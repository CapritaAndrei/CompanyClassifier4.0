"""
Decision Tracker for managing and persisting label curation decisions
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DecisionTracker:
    """Tracks and manages label curation decisions with undo/redo functionality"""
    
    def __init__(self, save_path: str = "label_curation_decisions.json"):
        """
        Initialize decision tracker
        
        Args:
            save_path: Path to save decisions file
        """
        self.save_path = Path(save_path)
        self.decisions = {}
        self.decision_history = []
        self.current_position = -1
        self.session_info = {
            'created_at': datetime.now().isoformat(),
            'last_updated': None,
            'version': '1.0'
        }
        
        # Load existing decisions if file exists
        self.load_decisions()
        
    def add_decision(self, sector: str, label: str, action: str, 
                    reason: str = "", affected_companies: int = 0,
                    metadata: Dict = None) -> bool:
        """
        Add a new decision
        
        Args:
            sector: Sector name
            label: Label name
            action: Action taken ('remove', 'keep', 'skip')
            reason: Reason for the decision
            affected_companies: Number of companies affected
            metadata: Additional metadata
            
        Returns:
            True if decision was added successfully
        """
        decision_id = f"{sector}:{label}"
        
        decision = {
            'sector': sector,
            'label': label,
            'action': action,
            'reason': reason,
            'affected_companies': affected_companies,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        # If we're not at the end of history, truncate the history
        if self.current_position < len(self.decision_history) - 1:
            self.decision_history = self.decision_history[:self.current_position + 1]
            
        # Add to history for undo/redo
        self.decision_history.append({
            'action': 'add',
            'decision_id': decision_id,
            'decision': decision.copy(),
            'previous_decision': self.decisions.get(decision_id)
        })
        
        # Update current decisions
        self.decisions[decision_id] = decision
        self.current_position += 1
        
        # Update session info
        self.session_info['last_updated'] = datetime.now().isoformat()
        
        logger.info(f"Added decision: {action} label '{label}' in sector '{sector}'")
        return True
        
    def undo_last_decision(self) -> bool:
        """
        Undo the last decision
        
        Returns:
            True if undo was successful
        """
        if self.current_position < 0:
            logger.warning("No decisions to undo")
            return False
            
        # Get the last decision from history
        last_action = self.decision_history[self.current_position]
        decision_id = last_action['decision_id']
        
        if last_action['action'] == 'add':
            # Restore previous decision or remove if it was new
            if last_action['previous_decision']:
                self.decisions[decision_id] = last_action['previous_decision']
            else:
                del self.decisions[decision_id]
                
        self.current_position -= 1
        
        logger.info(f"Undid decision for: {decision_id}")
        return True
        
    def redo_last_decision(self) -> bool:
        """
        Redo the last undone decision
        
        Returns:
            True if redo was successful
        """
        if self.current_position >= len(self.decision_history) - 1:
            logger.warning("No decisions to redo")
            return False
            
        self.current_position += 1
        
        # Get the decision to redo
        action = self.decision_history[self.current_position]
        decision_id = action['decision_id']
        
        if action['action'] == 'add':
            self.decisions[decision_id] = action['decision']
            
        logger.info(f"Redid decision for: {decision_id}")
        return True
        
    def get_decision(self, sector: str, label: str) -> Optional[Dict]:
        """
        Get decision for a specific sector-label combination
        
        Args:
            sector: Sector name
            label: Label name
            
        Returns:
            Decision dictionary or None if not found
        """
        decision_id = f"{sector}:{label}"
        return self.decisions.get(decision_id)
        
    def get_sector_decisions(self, sector: str) -> Dict[str, Dict]:
        """
        Get all decisions for a specific sector
        
        Args:
            sector: Sector name
            
        Returns:
            Dictionary mapping label to decision
        """
        sector_decisions = {}
        
        for decision_id, decision in self.decisions.items():
            if decision['sector'] == sector:
                sector_decisions[decision['label']] = decision
                
        return sector_decisions
        
    def get_decisions_by_action(self, action: str) -> Dict[str, Dict]:
        """
        Get all decisions with a specific action
        
        Args:
            action: Action type ('remove', 'keep', 'skip')
            
        Returns:
            Dictionary mapping decision_id to decision
        """
        filtered_decisions = {}
        
        for decision_id, decision in self.decisions.items():
            if decision['action'] == action:
                filtered_decisions[decision_id] = decision
                
        return filtered_decisions
        
    def get_decision_summary(self) -> Dict:
        """
        Get summary of all decisions
        
        Returns:
            Dictionary with decision statistics
        """
        action_counts = {'remove': 0, 'keep': 0, 'skip': 0}
        sector_counts = {}
        total_affected_companies = 0
        
        for decision in self.decisions.values():
            action = decision['action']
            sector = decision['sector']
            
            action_counts[action] += 1
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            total_affected_companies += decision['affected_companies']
            
        return {
            'total_decisions': len(self.decisions),
            'action_breakdown': action_counts,
            'sector_breakdown': sector_counts,
            'total_affected_companies': total_affected_companies,
            'can_undo': self.current_position >= 0,
            'can_redo': self.current_position < len(self.decision_history) - 1
        }
        
    def save_decisions(self) -> bool:
        """
        Save decisions to file
        
        Returns:
            True if save was successful
        """
        try:
            data = {
                'session_info': self.session_info,
                'decisions': self.decisions,
                'decision_history': self.decision_history,
                'current_position': self.current_position
            }
            
            with open(self.save_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved {len(self.decisions)} decisions to {self.save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save decisions: {e}")
            return False
            
    def load_decisions(self) -> bool:
        """
        Load decisions from file
        
        Returns:
            True if load was successful
        """
        if not self.save_path.exists():
            logger.info("No existing decisions file found")
            return True
            
        try:
            with open(self.save_path, 'r') as f:
                data = json.load(f)
                
            self.session_info = data.get('session_info', self.session_info)
            self.decisions = data.get('decisions', {})
            self.decision_history = data.get('decision_history', [])
            self.current_position = data.get('current_position', -1)
            
            logger.info(f"Loaded {len(self.decisions)} decisions from {self.save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load decisions: {e}")
            return False
            
    def export_decisions(self, export_path: str, format: str = 'json') -> bool:
        """
        Export decisions to different formats
        
        Args:
            export_path: Path to export file
            format: Export format ('json', 'csv', 'txt')
            
        Returns:
            True if export was successful
        """
        try:
            export_path = Path(export_path)
            
            if format == 'json':
                with open(export_path, 'w') as f:
                    json.dump(self.decisions, f, indent=2)
                    
            elif format == 'csv':
                import pandas as pd
                
                # Convert decisions to DataFrame
                rows = []
                for decision_id, decision in self.decisions.items():
                    row = decision.copy()
                    row['decision_id'] = decision_id
                    rows.append(row)
                    
                df = pd.DataFrame(rows)
                df.to_csv(export_path, index=False)
                
            elif format == 'txt':
                with open(export_path, 'w') as f:
                    f.write("Label Curation Decisions\n")
                    f.write("=" * 50 + "\n\n")
                    
                    summary = self.get_decision_summary()
                    f.write(f"Total decisions: {summary['total_decisions']}\n")
                    f.write(f"Actions: {summary['action_breakdown']}\n")
                    f.write(f"Sectors: {summary['sector_breakdown']}\n\n")
                    
                    for decision_id, decision in self.decisions.items():
                        f.write(f"Decision: {decision_id}\n")
                        f.write(f"  Action: {decision['action']}\n")
                        f.write(f"  Reason: {decision['reason']}\n")
                        f.write(f"  Affected companies: {decision['affected_companies']}\n")
                        f.write(f"  Timestamp: {decision['timestamp']}\n\n")
                        
            logger.info(f"Exported decisions to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export decisions: {e}")
            return False
            
    def clear_decisions(self) -> bool:
        """
        Clear all decisions (with confirmation)
        
        Returns:
            True if decisions were cleared
        """
        # Create backup before clearing
        backup_path = self.save_path.with_suffix('.backup.json')
        
        try:
            if self.save_path.exists():
                import shutil
                shutil.copy2(self.save_path, backup_path)
                
            self.decisions = {}
            self.decision_history = []
            self.current_position = -1
            self.session_info = {
                'created_at': datetime.now().isoformat(),
                'last_updated': None,
                'version': '1.0'
            }
            
            logger.info(f"Cleared all decisions (backup saved to {backup_path})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear decisions: {e}")
            return False
            
    def get_progress_info(self, total_labels: int) -> Dict:
        """
        Get progress information for current session
        
        Args:
            total_labels: Total number of labels to review
            
        Returns:
            Dictionary with progress information
        """
        decisions_made = len(self.decisions)
        progress_percentage = (decisions_made / total_labels * 100) if total_labels > 0 else 0
        
        return {
            'decisions_made': decisions_made,
            'total_labels': total_labels,
            'progress_percentage': progress_percentage,
            'remaining_labels': total_labels - decisions_made,
            'session_started': self.session_info['created_at'],
            'last_updated': self.session_info['last_updated']
        } 