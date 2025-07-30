#!/usr/bin/env python3
"""
MARS-GIS Project Progress Tracker Update Script

This script helps maintain the project progress tracker by providing
utilities to update task status, calculate completion percentages,
and generate progress reports.

Usage:
    python scripts/update_progress.py [command] [options]

Commands:
    status    - Show current project status
    update    - Update task status
    report    - Generate progress report
    metrics   - Display project metrics

Examples:
    python scripts/update_progress.py status
    python scripts/update_progress.py update SAFETY-007 complete
    python scripts/update_progress.py report weekly
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Task definitions from the progress tracker
TASKS = {
    # Phase 1: Foundation & Data Infrastructure
    "NASA-001": {
        "name": "NASA Mars Data Integration",
        "phase": 1,
        "priority": "critical",
        "status": "complete",
        "completion": 100,
        "owner": "Data Engineering Team",
        "dependencies": []
    },
    "USGS-002": {
        "name": "USGS Planetary Data Integration",
        "phase": 1,
        "priority": "critical", 
        "status": "complete",
        "completion": 100,
        "owner": "Data Engineering Team",
        "dependencies": []
    },
    "DB-003": {
        "name": "Geospatial Database Setup",
        "phase": 1,
        "priority": "critical",
        "status": "complete", 
        "completion": 100,
        "owner": "Backend Team",
        "dependencies": []
    },
    "STREAM-004": {
        "name": "Real-time Data Streaming",
        "phase": 1,
        "priority": "high",
        "status": "complete",
        "completion": 100,
        "owner": "Data Engineering Team", 
        "dependencies": []
    },
    "CLOUD-005": {
        "name": "Cloud Storage Architecture",
        "phase": 1,
        "priority": "high",
        "status": "complete",
        "completion": 100,
        "owner": "DevOps Team",
        "dependencies": []
    },
    
    # Phase 2: AI/ML Core Development
    "ML-006": {
        "name": "Terrain Classification Models",
        "phase": 2,
        "priority": "critical",
        "status": "complete",
        "completion": 100,
        "owner": "ML Engineering Team",
        "dependencies": ["DB-003"]
    },
    "SAFETY-007": {
        "name": "Landing Site Safety Assessment", 
        "phase": 2,
        "priority": "critical",
        "status": "in_progress",
        "completion": 75,
        "owner": "ML Engineering Team",
        "dependencies": ["ML-006"]
    },
    "ATMOS-008": {
        "name": "Atmospheric Analysis Models",
        "phase": 2, 
        "priority": "high",
        "status": "in_progress",
        "completion": 60,
        "owner": "ML Engineering Team",
        "dependencies": ["ML-006"]
    },
    "MLOPS-009": {
        "name": "MLOps Pipeline Implementation",
        "phase": 2,
        "priority": "high", 
        "status": "complete",
        "completion": 100,
        "owner": "ML Engineering Team",
        "dependencies": ["ML-006"]
    },
    "GPU-010": {
        "name": "GPU Computing Optimization",
        "phase": 2,
        "priority": "medium",
        "status": "not_started",
        "completion": 0,
        "owner": "ML Engineering Team",
        "dependencies": ["ML-006", "SAFETY-007"]
    },
    
    # Add more tasks as needed...
}

PHASES = {
    1: "Foundation & Data Infrastructure",
    2: "AI/ML Core Development", 
    3: "Geospatial Analysis Engine",
    4: "Visualization & User Interface",
    5: "Integration & Deployment"
}

STATUS_ICONS = {
    "complete": "✅",
    "in_progress": "🟡", 
    "not_started": "⭕",
    "blocked": "❌",
    "review": "🔄"
}

PRIORITY_ICONS = {
    "critical": "🔴",
    "high": "🟠",
    "medium": "🟡", 
    "low": "🟢"
}


def calculate_phase_completion(phase: int) -> float:
    """Calculate completion percentage for a specific phase."""
    phase_tasks = [task for task in TASKS.values() if task["phase"] == phase]
    if not phase_tasks:
        return 0.0
    
    total_completion = sum(task["completion"] for task in phase_tasks)
    return total_completion / len(phase_tasks)


def calculate_overall_completion() -> float:
    """Calculate overall project completion percentage."""
    if not TASKS:
        return 0.0
    
    total_completion = sum(task["completion"] for task in TASKS.values())
    return total_completion / len(TASKS)


def get_critical_path() -> List[str]:
    """Identify tasks on the critical path."""
    critical_tasks = []
    for task_id, task in TASKS.items():
        if task["priority"] == "critical" and task["status"] != "complete":
            critical_tasks.append(task_id)
    return critical_tasks


def show_status():
    """Display current project status."""
    print("🚀 MARS-GIS Project Status")
    print("=" * 50)
    
    overall = calculate_overall_completion()
    print(f"Overall Completion: {overall:.1f}%")
    print()
    
    for phase_num in range(1, 6):
        phase_completion = calculate_phase_completion(phase_num)
        phase_name = PHASES.get(phase_num, f"Phase {phase_num}")
        print(f"Phase {phase_num} ({phase_name}): {phase_completion:.1f}%")
    
    print()
    critical_path = get_critical_path()
    if critical_path:
        print("🚨 Critical Path Items:")
        for task_id in critical_path:
            task = TASKS[task_id]
            status_icon = STATUS_ICONS[task["status"]]
            priority_icon = PRIORITY_ICONS[task["priority"]]
            print(f"  {status_icon} {priority_icon} {task_id}: {task['name']} ({task['completion']}%)")
    else:
        print("✅ No critical path blockers")


def update_task_status(task_id: str, status: str, completion: Optional[int] = None):
    """Update the status of a specific task."""
    if task_id not in TASKS:
        print(f"❌ Task {task_id} not found")
        return False
    
    valid_statuses = ["complete", "in_progress", "not_started", "blocked", "review"]
    if status not in valid_statuses:
        print(f"❌ Invalid status. Valid options: {', '.join(valid_statuses)}")
        return False
    
    TASKS[task_id]["status"] = status
    
    if completion is not None:
        if 0 <= completion <= 100:
            TASKS[task_id]["completion"] = completion
        else:
            print("❌ Completion must be between 0 and 100")
            return False
    elif status == "complete":
        TASKS[task_id]["completion"] = 100
    elif status == "not_started":
        TASKS[task_id]["completion"] = 0
    
    print(f"✅ Updated {task_id}: {TASKS[task_id]['name']} -> {status}")
    if completion is not None:
        print(f"   Completion: {completion}%")
    
    return True


def generate_weekly_report():
    """Generate a weekly progress report."""
    print("📊 MARS-GIS Weekly Progress Report")
    print("=" * 50)
    print(f"Report Date: {datetime.now().strftime('%Y-%m-%d')}")
    print()
    
    # Summary statistics
    total_tasks = len(TASKS)
    completed = len([t for t in TASKS.values() if t["status"] == "complete"])
    in_progress = len([t for t in TASKS.values() if t["status"] == "in_progress"])
    not_started = len([t for t in TASKS.values() if t["status"] == "not_started"])
    
    print(f"Task Summary:")
    print(f"  Total Tasks: {total_tasks}")
    print(f"  ✅ Completed: {completed}")
    print(f"  🟡 In Progress: {in_progress}")
    print(f"  ⭕ Not Started: {not_started}")
    print(f"  Overall Progress: {calculate_overall_completion():.1f}%")
    print()
    
    # Phase breakdown
    print("Phase Progress:")
    for phase_num in range(1, 6):
        completion = calculate_phase_completion(phase_num)
        phase_name = PHASES.get(phase_num, f"Phase {phase_num}")
        bar_length = int(completion / 2)  # Scale to 50 chars max
        bar = "█" * bar_length + "▓" * (50 - bar_length)
        print(f"  Phase {phase_num}: {bar} {completion:.1f}%")
    print()
    
    # This week's focus
    print("🎯 This Week's Focus:")
    focus_tasks = [
        task_id for task_id, task in TASKS.items() 
        if task["status"] == "in_progress" and task["priority"] in ["critical", "high"]
    ]
    for task_id in focus_tasks[:5]:  # Top 5 focus items
        task = TASKS[task_id]
        priority_icon = PRIORITY_ICONS[task["priority"]]
        print(f"  {priority_icon} {task_id}: {task['name']} ({task['completion']}% complete)")


def show_metrics():
    """Display detailed project metrics."""
    print("📈 MARS-GIS Project Metrics")
    print("=" * 50)
    
    # Completion metrics
    overall = calculate_overall_completion()
    print(f"Overall Completion: {overall:.1f}%")
    
    # Velocity calculation (tasks completed per week)
    # This is a simplified calculation - in practice you'd track completion dates
    completed_tasks = len([t for t in TASKS.values() if t["status"] == "complete"])
    weeks_elapsed = 8  # Approximate based on project timeline
    velocity = completed_tasks / weeks_elapsed if weeks_elapsed > 0 else 0
    print(f"Velocity: {velocity:.1f} tasks/week")
    
    # Priority distribution
    priority_counts = {}
    for task in TASKS.values():
        priority = task["priority"]
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
    
    print("\nPriority Distribution:")
    for priority, count in priority_counts.items():
        icon = PRIORITY_ICONS[priority]
        print(f"  {icon} {priority.title()}: {count} tasks")
    
    # Status distribution
    status_counts = {}
    for task in TASKS.values():
        status = task["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    
    print("\nStatus Distribution:")
    for status, count in status_counts.items():
        icon = STATUS_ICONS[status]
        print(f"  {icon} {status.replace('_', ' ').title()}: {count} tasks")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="MARS-GIS Project Progress Tracker")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Status command
    subparsers.add_parser("status", help="Show current project status")
    
    # Update command
    update_parser = subparsers.add_parser("update", help="Update task status")
    update_parser.add_argument("task_id", help="Task ID to update")
    update_parser.add_argument("status", help="New status")
    update_parser.add_argument("--completion", type=int, help="Completion percentage (0-100)")
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate progress report")
    report_parser.add_argument("type", choices=["weekly"], default="weekly", nargs="?")
    
    # Metrics command
    subparsers.add_parser("metrics", help="Display project metrics")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "status":
        show_status()
    elif args.command == "update":
        update_task_status(args.task_id, args.status, args.completion)
    elif args.command == "report":
        if args.type == "weekly":
            generate_weekly_report()
    elif args.command == "metrics":
        show_metrics()


if __name__ == "__main__":
    main()
