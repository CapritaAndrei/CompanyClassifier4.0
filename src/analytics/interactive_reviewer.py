"""
Interactive Reviewer for label curation with comprehensive context and decision tracking
"""

import sys
from typing import Dict, List, Optional, Tuple
import logging

from .heatmap_analyzer import HeatmapAnalyzer
from .smart_sampler import SmartCompanySampler
from .context_provider import ContextProvider
from .decision_tracker import DecisionTracker

logger = logging.getLogger(__name__)

class InteractiveReviewer:
    """Main interactive reviewer for label curation"""
    
    def __init__(self, company_df, company_tags, company_metadata, 
                 classification_data, decision_save_path: str = "label_curation_decisions.json"):
        """
        Initialize interactive reviewer
        
        Args:
            company_df: DataFrame with company information
            company_tags: Dict mapping company_id to business tags
            company_metadata: Dict mapping company_id to metadata
            classification_data: Dict mapping company_id to classification results
            decision_save_path: Path to save decisions
        """
        self.analyzer = HeatmapAnalyzer()
        self.sampler = SmartCompanySampler(company_metadata)
        self.context_provider = ContextProvider(company_df, company_tags, classification_data)
        self.decision_tracker = DecisionTracker(decision_save_path)
        
        # Load data into analyzer
        self.analyzer.load_classification_data(classification_data, company_metadata)
        
        logger.info("Interactive reviewer initialized")
        
    def start_review_session(self) -> None:
        """Start the interactive review session"""
        
        print("\n" + "=" * 80)
        print("🎯 INSURANCE LABEL CURATION SYSTEM")
        print("=" * 80)
        
        # Show overview statistics
        overview = self.analyzer.get_overview_statistics()
        print(f"\nOverview Statistics:")
        print(f"  Total companies: {overview['total_companies']:,}")
        print(f"  Companies with labels: {overview['companies_with_labels']:,}")
        print(f"  Total classifications: {overview['total_classifications']:,}")
        print(f"  Unique labels: {overview['unique_labels']:,}")
        print(f"  Sectors: {overview['sectors_count']}")
        
        # Show existing decisions
        decision_summary = self.decision_tracker.get_decision_summary()
        if decision_summary['total_decisions'] > 0:
            print(f"\n📝 Existing Decisions:")
            print(f"  Total decisions: {decision_summary['total_decisions']}")
            print(f"  Actions: {decision_summary['action_breakdown']}")
            print(f"  Can undo: {decision_summary['can_undo']}")
            print(f"  Can redo: {decision_summary['can_redo']}")
        
        # Main menu
        while True:
            choice = self._show_main_menu()
            
            if choice == '1':
                self._sector_review_workflow()
            elif choice == '2':
                self._show_decision_summary()
            elif choice == '3':
                self._manage_decisions()
            elif choice == '4':
                self._export_options()
            elif choice == '5':
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please try again.")
                
    def _show_main_menu(self) -> str:
        """Show main menu and get user choice"""
        
        print("\n" + "=" * 50)
        print("📋 MAIN MENU")
        print("=" * 50)
        print("1. 🔍 Review labels by sector")
        print("2. 📊 View decision summary")
        print("3. 🔧 Manage decisions (undo/redo/clear)")
        print("4. 📤 Export options")
        print("5. 🚪 Exit")
        
        return input("\nSelect option (1-5): ").strip()
        
    def _sector_review_workflow(self) -> None:
        """Workflow for reviewing labels by sector"""
        
        sectors = self.analyzer.get_sectors()
        
        print("\n" + "=" * 50)
        print("🏢 SECTOR SELECTION")
        print("=" * 50)
        
        for i, sector in enumerate(sectors, 1):
            sector_summary = self.analyzer.get_sector_summary(sector)
            print(f"{i}. {sector} ({sector_summary['total_companies']} companies, "
                  f"{sector_summary['total_labels']} labels)")
        
        while True:
            try:
                choice = input(f"\nSelect sector (1-{len(sectors)}) or 'back': ").strip()
                
                if choice.lower() == 'back':
                    return
                    
                sector_idx = int(choice) - 1
                if 0 <= sector_idx < len(sectors):
                    selected_sector = sectors[sector_idx]
                    self._review_sector_labels(selected_sector)
                    break
                else:
                    print("❌ Invalid sector number")
                    
            except ValueError:
                print("❌ Please enter a valid number")
                
    def _review_sector_labels(self, sector: str) -> None:
        """Review labels for a specific sector"""
        
        print(f"\n" + "=" * 60)
        print(f"🔍 REVIEWING SECTOR: {sector}")
        print("=" * 60)
        
        # Get sector summary
        sector_summary = self.analyzer.get_sector_summary(sector)
        print(f"Sector Summary:")
        print(f"  Companies: {sector_summary['total_companies']}")
        print(f"  Labels: {sector_summary['total_labels']}")
        print(f"  Categories: {sector_summary['categories_count']}")
        print(f"  Niches: {sector_summary['niches_count']}")
        
        # Get labels sorted by frequency (rarest first)
        labels = self.analyzer.get_sector_labels_by_frequency(sector, ascending=True)
        
        print(f"\nReviewing {len(labels)} labels (starting with rarest)...")
        
        for i, (label, count) in enumerate(labels, 1):
            # Check if we already have a decision for this label
            existing_decision = self.decision_tracker.get_decision(sector, label)
            
            if existing_decision:
                print(f"\n⏭️  Skipping {label} (already decided: {existing_decision['action']})")
                continue
                
            print(f"\n" + "=" * 50)
            print(f"📝 LABEL {i}/{len(labels)}: {label}")
            print("=" * 50)
            
            # Show label analytics
            analytics = self.analyzer.get_label_analytics(sector, label)
            self._show_label_analytics(analytics)
            
            # Show diverse company samples
            company_ids = self.analyzer.get_companies_with_label(sector, label)
            diverse_samples = self.sampler.get_diverse_samples(company_ids, max_samples=10)
            
            print(f"\n📋 COMPANY SAMPLES ({len(diverse_samples)} diverse examples):")
            print("-" * 50)
            
            for j, company_id in enumerate(diverse_samples, 1):
                context = self.context_provider.format_company_display(company_id, label)
                print(f"\n{j}. {context}")
                
            # Get user decision
            decision = self._get_user_decision(sector, label, len(company_ids))
            
            if decision == 'quit':
                print("\n💾 Saving progress...")
                self.decision_tracker.save_decisions()
                return
                
    def _show_label_analytics(self, analytics: Dict) -> None:
        """Show analytics for a label"""
        
        if not analytics:
            print("⚠️  No analytics available")
            return
            
        print(f"📊 Label Analytics:")
        print(f"  Frequency: {analytics['total_count']} companies")
        print(f"  Categories spread: {len(analytics['categories'])} categories")
        print(f"  Niches spread: {len(analytics['niches'])} niches")
        
        if 'confidence_stats' in analytics:
            stats = analytics['confidence_stats']
            print(f"  Confidence: {stats['min']:.3f} - {stats['max']:.3f} "
                  f"(avg: {stats['avg']:.3f})")
        
        # Show category distribution
        if analytics['categories']:
            print(f"  Top categories: {', '.join(list(analytics['categories'].keys())[:3])}")
            
    def _get_user_decision(self, sector: str, label: str, affected_companies: int) -> str:
        """Get user decision for a label"""
        
        print(f"\n🤔 DECISION TIME:")
        print(f"What would you like to do with '{label}' in {sector}?")
        print("  [r] Remove - Delete this label completely")
        print("  [k] Keep - Keep this label")
        print("  [s] Skip - Skip for now (can review later)")
        print("  [u] Undo - Undo last decision")
        print("  [i] Info - Show more detailed company info")
        print("  [q] Quit - Save and exit")
        
        while True:
            choice = input("\nYour choice (r/k/s/u/i/q): ").strip().lower()
            
            if choice in ['r', 'remove']:
                reason = input("Reason for removal (optional): ").strip()
                self.decision_tracker.add_decision(
                    sector, label, 'remove', reason, affected_companies
                )
                print(f"✅ Marked '{label}' for removal")
                return 'remove'
                
            elif choice in ['k', 'keep']:
                reason = input("Reason for keeping (optional): ").strip()
                self.decision_tracker.add_decision(
                    sector, label, 'keep', reason, affected_companies
                )
                print(f"✅ Marked '{label}' to keep")
                return 'keep'
                
            elif choice in ['s', 'skip']:
                self.decision_tracker.add_decision(
                    sector, label, 'skip', "Skipped for later review", affected_companies
                )
                print(f"⏭️  Skipped '{label}'")
                return 'skip'
                
            elif choice in ['u', 'undo']:
                if self.decision_tracker.undo_last_decision():
                    print("↩️  Undid last decision")
                else:
                    print("❌ No decisions to undo")
                    
            elif choice in ['i', 'info']:
                self._show_detailed_info(sector, label)
                
            elif choice in ['q', 'quit']:
                return 'quit'
                
            else:
                print("❌ Invalid choice. Please try again.")
                
    def _show_detailed_info(self, sector: str, label: str) -> None:
        """Show detailed information for a label"""
        
        print(f"\n" + "=" * 60)
        print(f"📋 DETAILED INFO: {label} in {sector}")
        print("=" * 60)
        
        # Get all companies with this label
        company_ids = self.analyzer.get_companies_with_label(sector, label)
        
        print(f"Total companies with this label: {len(company_ids)}")
        
        # Show diversity stats
        diversity_stats = self.sampler.get_diversity_stats(company_ids)
        print(f"Diversity:")
        print(f"  Categories: {diversity_stats['unique_categories']} "
              f"({', '.join(diversity_stats['categories'][:5])})")
        print(f"  Niches: {diversity_stats['unique_niches']}")
        
        # Show detailed company breakdown
        breakdown = self.sampler.get_category_niche_breakdown(company_ids)
        print(f"\nCategory-Niche Breakdown:")
        for category, niches in breakdown.items():
            print(f"  {category}: {len(niches)} niches")
            
        input("\nPress Enter to continue...")
        
    def _show_decision_summary(self) -> None:
        """Show summary of all decisions made"""
        
        print(f"\n" + "=" * 60)
        print("📊 DECISION SUMMARY")
        print("=" * 60)
        
        summary = self.decision_tracker.get_decision_summary()
        
        print(f"Total decisions: {summary['total_decisions']}")
        print(f"Actions: {summary['action_breakdown']}")
        print(f"Affected companies: {summary['total_affected_companies']}")
        print(f"Sectors reviewed: {summary['sector_breakdown']}")
        
        if summary['total_decisions'] > 0:
            print(f"\nRecent decisions:")
            # Show last 5 decisions
            recent_decisions = list(self.decision_tracker.decisions.items())[-5:]
            for decision_id, decision in recent_decisions:
                print(f"  {decision['action']}: {decision['label']} "
                      f"({decision['sector']}) - {decision['reason']}")
                      
        input("\nPress Enter to continue...")
        
    def _manage_decisions(self) -> None:
        """Manage decisions (undo/redo/clear)"""
        
        print(f"\n" + "=" * 50)
        print("🔧 DECISION MANAGEMENT")
        print("=" * 50)
        
        summary = self.decision_tracker.get_decision_summary()
        
        print("Options:")
        print(f"1. Undo last decision (available: {summary['can_undo']})")
        print(f"2. Redo last decision (available: {summary['can_redo']})")
        print("3. Clear all decisions")
        print("4. Save decisions")
        print("5. Back to main menu")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            if self.decision_tracker.undo_last_decision():
                print("✅ Undid last decision")
            else:
                print("❌ No decisions to undo")
                
        elif choice == '2':
            if self.decision_tracker.redo_last_decision():
                print("✅ Redid last decision")
            else:
                print("❌ No decisions to redo")
                
        elif choice == '3':
            confirm = input("⚠️  Are you sure you want to clear all decisions? (yes/no): ")
            if confirm.lower() == 'yes':
                self.decision_tracker.clear_decisions()
                print("✅ All decisions cleared")
            else:
                print("❌ Clear cancelled")
                
        elif choice == '4':
            self.decision_tracker.save_decisions()
            print("✅ Decisions saved")
            
        elif choice == '5':
            return
            
        input("\nPress Enter to continue...")
        
    def _export_options(self) -> None:
        """Show export options"""
        
        print(f"\n" + "=" * 50)
        print("📤 EXPORT OPTIONS")
        print("=" * 50)
        
        print("1. Export decisions to JSON")
        print("2. Export decisions to CSV")
        print("3. Export heatmap data")
        print("4. Back to main menu")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            path = input("Enter export path (default: decisions.json): ").strip()
            if not path:
                path = "decisions.json"
            self.decision_tracker.export_decisions(path, 'json')
            print(f"✅ Exported to {path}")
            
        elif choice == '2':
            path = input("Enter export path (default: decisions.csv): ").strip()
            if not path:
                path = "decisions.csv"
            self.decision_tracker.export_decisions(path, 'csv')
            print(f"✅ Exported to {path}")
            
        elif choice == '3':
            path = input("Enter export path (default: heatmap.csv): ").strip()
            if not path:
                path = "heatmap.csv"
            self.analyzer.export_heatmap_data(path)
            print(f"✅ Exported to {path}")
            
        elif choice == '4':
            return
            
        input("\nPress Enter to continue...")
        
    def get_decisions_for_application(self) -> Dict[str, List[str]]:
        """
        Get decisions formatted for application to classification system
        
        Returns:
            Dictionary with labels to remove by sector
        """
        labels_to_remove = {}
        
        removed_decisions = self.decision_tracker.get_decisions_by_action('remove')
        
        for decision_id, decision in removed_decisions.items():
            sector = decision['sector']
            label = decision['label']
            
            if sector not in labels_to_remove:
                labels_to_remove[sector] = []
                
            labels_to_remove[sector].append(label)
            
        return labels_to_remove 