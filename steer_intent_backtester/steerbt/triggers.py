"""
Rebalancing triggers for CLMM strategies.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class TriggerBase:
    """Base class for rebalancing triggers."""
    
    def __init__(self, **kwargs):
        self.last_triggered = None
        self.trigger_count = 0
    
    def should_trigger(self, current_state: Dict[str, Any]) -> bool:
        """
        Determine if trigger should fire.
        
        Args:
            current_state: Current market and position state
            
        Returns:
            True if trigger should fire
        """
        raise NotImplementedError
    
    def reset(self):
        """Reset trigger state."""
        self.last_triggered = None
        self.trigger_count = 0


class GapFromCenterTrigger(TriggerBase):
    """Trigger when price moves a certain distance from position center."""
    
    def __init__(self, gap_ticks: int = 100, gap_bps: Optional[float] = None):
        super().__init__()
        self.gap_ticks = gap_ticks
        self.gap_bps = gap_bps
    
    def should_trigger(self, current_state: Dict[str, Any]) -> bool:
        """
        Check if price has moved too far from position center.
        
        Args:
            current_state: Must contain 'current_price', 'position_center'
            
        Returns:
            True if gap threshold exceeded
        """
        current_price = current_state["current_price"]
        position_center = current_state["position_center"]
        
        if self.gap_bps is not None:
            # Use basis points
            gap_pct = abs(current_price - position_center) / position_center
            return gap_pct * 10000 >= self.gap_bps
        else:
            # Use tick distance (simplified)
            price_diff = abs(current_price - position_center)
            return price_diff >= self.gap_ticks * 0.0001  # Approximate tick size


class RangeInactiveTrigger(TriggerBase):
    """Trigger when price is outside position range."""
    
    def __init__(self):
        super().__init__()
    
    def should_trigger(self, current_state: Dict[str, Any]) -> bool:
        """
        Check if price is outside position range.
        
        Args:
            current_state: Must contain 'current_price', 'lower_price', 'upper_price'
            
        Returns:
            True if price outside range
        """
        current_price = current_state["current_price"]
        lower_price = current_state["lower_price"]
        upper_price = current_state["upper_price"]
        
        return current_price < lower_price or current_price > upper_price


class PercentDriftTrigger(TriggerBase):
    """Trigger when position drifts by a certain percentage."""
    
    def __init__(self, drift_threshold_pct: float = 5.0):
        super().__init__()
        self.drift_threshold_pct = drift_threshold_pct
        self.last_position_value = None
    
    def should_trigger(self, current_state: Dict[str, Any]) -> bool:
        """
        Check if position has drifted by threshold percentage.
        
        Args:
            current_state: Must contain 'position_value'
            
        Returns:
            True if drift threshold exceeded
        """
        current_position_value = current_state["position_value"]
        
        if self.last_position_value is None:
            self.last_position_value = current_position_value
            return False
        
        drift_pct = abs(current_position_value - self.last_position_value) / self.last_position_value * 100
        
        if drift_pct >= self.drift_threshold_pct:
            self.last_position_value = current_position_value
            return True
        
        return False


class OneWayExitTrigger(TriggerBase):
    """Trigger when price moves in one direction for extended period."""
    
    def __init__(self, direction: str = "down", consecutive_bars: int = 5):
        super().__init__()
        self.direction = direction
        self.consecutive_bars = consecutive_bars
        self.consecutive_count = 0
        self.last_price = None
    
    def should_trigger(self, current_state: Dict[str, Any]) -> bool:
        """
        Check if price has moved in one direction for consecutive bars.
        
        Args:
            current_state: Must contain 'current_price'
            
        Returns:
            True if consecutive movement threshold met
        """
        current_price = current_state["current_price"]
        
        if self.last_price is None:
            self.last_price = current_price
            return False
        
        if self.direction == "up" and current_price > self.last_price:
            self.consecutive_count += 1
        elif self.direction == "down" and current_price < self.last_price:
            self.consecutive_count += 1
        else:
            self.consecutive_count = 0
        
        self.last_price = current_price
        
        return self.consecutive_count >= self.consecutive_bars


class ElapsedTimeTrigger(TriggerBase):
    """Trigger after a certain time has elapsed."""
    
    def __init__(self, time_delta: timedelta):
        super().__init__()
        self.time_delta = time_delta
        self.last_triggered = None
    
    def should_trigger(self, current_state: Dict[str, Any]) -> bool:
        """
        Check if enough time has elapsed since last trigger.
        
        Args:
            current_state: Must contain 'current_timestamp'
            
        Returns:
            True if time threshold exceeded
        """
        current_timestamp = current_state["current_timestamp"]
        
        if self.last_triggered is None:
            self.last_triggered = current_timestamp
            return False
        
        time_elapsed = current_timestamp - self.last_triggered
        
        return time_elapsed >= self.time_delta


class VolatilityTrigger(TriggerBase):
    """Trigger based on volatility threshold."""
    
    def __init__(self, volatility_threshold: float = 0.02, lookback_periods: int = 20):
        super().__init__()
        self.volatility_threshold = volatility_threshold
        self.lookback_periods = lookback_periods
        self.price_history = []
    
    def should_trigger(self, current_state: Dict[str, Any]) -> bool:
        """
        Check if volatility exceeds threshold.
        
        Args:
            current_state: Must contain 'current_price'
            
        Returns:
            True if volatility threshold exceeded
        """
        current_price = current_state["current_price"]
        self.price_history.append(current_price)
        
        # Keep only lookback periods
        if len(self.price_history) > self.lookback_periods:
            self.price_history.pop(0)
        
        if len(self.price_history) < self.lookback_periods:
            return False
        
        # Calculate rolling volatility
        returns = np.diff(np.log(self.price_history))
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        return volatility >= self.volatility_threshold


class CompositeTrigger(TriggerBase):
    """Combines multiple triggers with logical operators."""
    
    def __init__(self, triggers: list, operator: str = "OR"):
        super().__init__()
        self.triggers = triggers
        self.operator = operator.upper()
        
        if self.operator not in ["AND", "OR"]:
            raise ValueError("Operator must be 'AND' or 'OR'")
    
    def should_trigger(self, current_state: Dict[str, Any]) -> bool:
        """
        Evaluate composite trigger based on operator.
        
        Args:
            current_state: Current market and position state
            
        Returns:
            True if composite trigger should fire
        """
        trigger_results = [trigger.should_trigger(current_state) for trigger in self.triggers]
        
        if self.operator == "AND":
            return all(trigger_results)
        else:  # OR
            return any(trigger_results)
    
    def reset(self):
        """Reset all triggers."""
        super().reset()
        for trigger in self.triggers:
            trigger.reset()


class TriggerManager:
    """Manages multiple triggers and their execution."""
    
    def __init__(self):
        self.triggers: Dict[str, TriggerBase] = {}
    
    def add_trigger(self, name: str, trigger: TriggerBase):
        """Add a trigger to the manager."""
        self.triggers[name] = trigger
    
    def remove_trigger(self, name: str):
        """Remove a trigger from the manager."""
        if name in self.triggers:
            del self.triggers[name]
    
    def should_trigger_any(self, current_state: Dict[str, Any]) -> Tuple[bool, list]:
        """
        Check if any trigger should fire.
        
        Args:
            current_state: Current market and position state
            
        Returns:
            Tuple of (should_trigger, list_of_triggered_names)
        """
        triggered_names = []
        
        for name, trigger in self.triggers.items():
            if trigger.should_trigger(current_state):
                triggered_names.append(name)
                trigger.last_triggered = current_state.get("current_timestamp")
                trigger.trigger_count += 1
        
        return len(triggered_names) > 0, triggered_names
    
    def reset_all(self):
        """Reset all triggers."""
        for trigger in self.triggers.values():
            trigger.reset()
    
    def get_trigger_stats(self) -> Dict[str, Dict]:
        """Get statistics for all triggers."""
        stats = {}
        
        for name, trigger in self.triggers.items():
            stats[name] = {
                "trigger_count": trigger.trigger_count,
                "last_triggered": trigger.last_triggered
            }
        
        return stats
