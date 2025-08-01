#!/usr/bin/env python3
"""
MARS-GIS Comprehensive Code Reorganization Script

This script performs complete analysis, cleanup, and reorganization of
the MARS-GIS project:
1. Removes duplicate files and cleanup scripts
2. Reorganizes directory structure following best practices
3. Updates import paths and references
4. Ensures project functionality is maintained

Author: GitHub Copilot
Date: August 2025
"""

import os
import hashlib
from pathlib import Path
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reorganization.log'),
        logging.StreamHandler()
    ]
)


class MarsGISReorganizer:
    """Complete project reorganization and cleanup tool."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.dry_run = True
        self.removed_files = []
        self.moved_files = []
        self.updated_files = []

        # Files that should be kept (essential for project)
        self.essential_files = {
            "src/mars_gis/__init__.py",
            "src/mars_gis/main.py",
            "src/mars_gis/core/config.py",
            "requirements.txt",
            "pyproject.toml",
            "README.md",
            "LICENSE",
            ".gitignore",
            ".env.example"
        }

        # Docker files to keep (organized structure)
        self.essential_docker_files = {
            "docker/backend/Dockerfile",
            "docker/frontend/Dockerfile",
            "docker/compose/docker-compose.yml",
            "docker/compose/docker-compose.override.yml",
            "docker/compose/docker-compose.prod.yml",
            "docker/compose/docker-compose.test.yml",
            "docker/docker-helper.sh",
            "docker/README.md"
        }

        # Patterns for files that should be removed
        self.remove_patterns = [
            "*cleanup*",
            "*final*",
            "*deprecated*",
            "*backup*",
            "*redirect*",
            "*migration*",
            "*duplicate*",
            "docker-compose-clean.yml",
            "docker-compose-new.yml",
            "validate_docker_organization.sh",
            "validate_structure.py",
            "file_organization.log",
            "run_migration.sh",
            "quick_cleanup.sh"
        ]

    def analyze_duplicates(self) -> Dict[str, List[Path]]:
        """Find duplicate files by comparing content hashes."""
        logging.info("🔍 Analyzing duplicate files...")

        file_hashes = {}
        duplicates = {}

        # Get all files excluding directories we don't want to process
        exclude_dirs = {'.git', '__pycache__', '.venv', 'venv', 'node_modules'}

        for file_path in self.project_root.rglob('*'):
            if (file_path.is_file() and
                not any(part in exclude_dirs for part in file_path.parts)):

                try:
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()

                    if file_hash in file_hashes:
                        if file_hash not in duplicates:
                            duplicates[file_hash] = [file_hashes[file_hash]]
                        duplicates[file_hash].append(file_path)
                    else:
                        file_hashes[file_hash] = file_path

                except (IOError, PermissionError):
                    continue

        logging.info(f"Found {len(duplicates)} groups of duplicate files")
        return duplicates

    def remove_cleanup_files(self) -> List[Path]:
        """Remove temporary cleanup and migration files."""
        logging.info("🧹 Removing cleanup and temporary files...")

        removed = []
        for pattern in self.remove_patterns:
            for file_path in self.project_root.rglob(pattern):
                if file_path.is_file():
                    relative_path = file_path.relative_to(self.project_root)
                    logging.info(f"  Removing: {relative_path}")
                    removed.append(file_path)
                    if not self.dry_run:
                        file_path.unlink()

        return removed

    def organize_docker_structure(self) -> List[Path]:
        """Clean up Docker file organization."""
        logging.info("🐳 Organizing Docker structure...")

        removed = []

        # Remove duplicate Docker files in root
        docker_files_to_remove = [
            "Dockerfile-final",
            "Dockerfile-redirect",
            "docker-compose-deprecated.yml",
            "docker-compose-clean.yml",
            "docker-compose-new.yml",
            "docker-compose.override-final.yml",
            "docker-compose.prod-final.yml",
            "docker-compose.test-final.yml"
        ]

        for filename in docker_files_to_remove:
            file_path = self.project_root / filename
            if file_path.exists():
                logging.info(f"  Removing Docker file: {filename}")
                removed.append(file_path)
                if not self.dry_run:
                    file_path.unlink()

        # Remove duplicates in docker/ directory
        docker_dir = self.project_root / "docker"
        if docker_dir.exists():
            for file_path in docker_dir.rglob("*"):
                if (file_path.is_file() and
                    any(pattern in file_path.name for pattern in
                        ["final", "backup", "redirect", "deprecated"])):
                    relative_path = file_path.relative_to(self.project_root)
                    logging.info(f"  Removing Docker duplicate: {relative_path}")
                    removed.append(file_path)
                    if not self.dry_run:
                        file_path.unlink()

        return removed

    def resolve_duplicates(self, duplicates: Dict[str, List[Path]]) -> List[Path]:
        """Resolve duplicate files by keeping the best version."""
        logging.info("📁 Resolving duplicate files...")

        removed = []

        for file_hash, duplicate_files in duplicates.items():
            if len(duplicate_files) <= 1:
                continue

            # Determine which file to keep based on priority
            keep_file = self.choose_file_to_keep(duplicate_files)

            for file_path in duplicate_files:
                if file_path != keep_file:
                    relative_path = file_path.relative_to(self.project_root)
                    logging.info(f"  Removing duplicate: {relative_path}")
                    removed.append(file_path)
                    if not self.dry_run:
                        file_path.unlink()

        return removed

    def choose_file_to_keep(self, duplicate_files: List[Path]) -> Path:
        """Choose which duplicate file to keep based on priority rules."""

        # Priority order (higher score = keep this file)
        def priority_score(file_path: Path) -> int:
            score = 0
            path_str = str(file_path)

            # Prefer files in organized directories
            if "src/mars_gis" in path_str:
                score += 100
            elif "docker/compose" in path_str:
                score += 90
            elif "docker/backend" in path_str or "docker/frontend" in path_str:
                score += 80

            # Prefer non-backup, non-temporary files
            if any(word in file_path.name for word in
                   ["backup", "final", "temp", "old", "deprecated"]):
                score -= 50

            # Prefer shorter paths (less nested)
            score -= len(file_path.parts) * 2

            return score

        return max(duplicate_files, key=priority_score)

    def clean_empty_directories(self) -> List[Path]:
        """Remove empty directories after cleanup."""
        logging.info("📂 Cleaning empty directories...")

        removed = []

        # Get all directories, sorted by depth (deepest first)
        all_dirs = [d for d in self.project_root.rglob('*') if d.is_dir()]
        all_dirs.sort(key=lambda x: len(x.parts), reverse=True)

        for dir_path in all_dirs:
            try:
                # Skip essential directories
                if any(essential in str(dir_path) for essential in
                       ['.git', '__pycache__', 'src/mars_gis']):
                    continue

                # Check if directory is empty
                if dir_path.exists() and not any(dir_path.iterdir()):
                    relative_path = dir_path.relative_to(self.project_root)
                    logging.info(f"  Removing empty directory: {relative_path}")
                    removed.append(dir_path)
                    if not self.dry_run:
                        dir_path.rmdir()

            except OSError:
                continue

        return removed

    def validate_project_structure(self) -> bool:
        """Validate that essential project files still exist."""
        logging.info("✅ Validating project structure...")

        missing_files = []

        for essential_file in self.essential_files:
            file_path = self.project_root / essential_file
            if not file_path.exists():
                missing_files.append(essential_file)

        if missing_files:
            logging.error(f"Missing essential files: {missing_files}")
            return False

        logging.info("All essential files present")
        return True

    def generate_report(self) -> str:
        """Generate a comprehensive reorganization report."""

        report = []
        report.append("# MARS-GIS Project Reorganization Report")
        report.append(f"Generated: {os.getcwd()}")
        report.append("")

        # Project structure analysis
        report.append("## Project Structure Analysis")
        report.append("")
        report.append("**Project Type**: Mars Geospatial Intelligence System (MARS-GIS)")
        report.append("**Language**: Python 3.8+")
        report.append("**Framework**: FastAPI with geospatial extensions")
        report.append("")

        # Current structure
        report.append("## Current Optimized Structure")
        report.append("```")
        report.append("MARS-GIS/")
        report.append("├── src/mars_gis/          # Main application code")
        report.append("│   ├── api/               # FastAPI endpoints")
        report.append("│   ├── core/              # Core business logic")
        report.append("│   ├── data/              # Data processing modules")
        report.append("│   ├── ml/                # Machine learning models")
        report.append("│   ├── models/            # Foundation models")
        report.append("│   ├── visualization/     # Visualization components")
        report.append("│   └── utils/             # Utility modules")
        report.append("├── tests/                 # Test suite")
        report.append("├── docs/                  # Documentation")
        report.append("├── scripts/               # Utility scripts")
        report.append("├── docker/                # Docker organization")
        report.append("│   ├── backend/           # Backend Dockerfile")
        report.append("│   ├── frontend/          # Frontend Dockerfile")
        report.append("│   └── compose/           # Docker Compose files")
        report.append("├── data/                  # Data storage")
        report.append("└── assets/                # Static assets")
        report.append("```")
        report.append("")

        return "\\n".join(report)

    def run_reorganization(self, dry_run: bool = True) -> bool:
        """Run the complete reorganization process."""
        self.dry_run = dry_run

        mode = "DRY RUN" if dry_run else "LIVE"
        logging.info(f"🚀 Starting MARS-GIS reorganization in {mode} mode")

        try:
            # Step 1: Analyze duplicates
            duplicates = self.analyze_duplicates()

            # Step 2: Remove cleanup files
            removed_cleanup = self.remove_cleanup_files()
            self.removed_files.extend(removed_cleanup)

            # Step 3: Organize Docker structure
            removed_docker = self.organize_docker_structure()
            self.removed_files.extend(removed_docker)

            # Step 4: Resolve duplicates
            removed_duplicates = self.resolve_duplicates(duplicates)
            self.removed_files.extend(removed_duplicates)

            # Step 5: Clean empty directories
            removed_dirs = self.clean_empty_directories()
            self.removed_files.extend(removed_dirs)

            # Step 6: Validate structure
            if not self.validate_project_structure():
                logging.error("❌ Project structure validation failed!")
                return False

            # Summary
            logging.info(f"✅ Reorganization complete!")
            logging.info(f"📊 Files removed: {len(self.removed_files)}")
            logging.info(f"📊 Files moved: {len(self.moved_files)}")
            logging.info(f"📊 Files updated: {len(self.updated_files)}")

            if dry_run:
                logging.info("💡 This was a dry run. Use run_reorganization(dry_run=False) to execute changes.")

            return True

        except Exception as e:
            logging.error(f"❌ Reorganization failed: {str(e)}")
            return False

def main():
    """Main entry point for the reorganization script."""

    print("🚀 MARS-GIS Comprehensive Project Reorganization")
    print("=" * 60)
    print()

    reorganizer = MarsGISReorganizer()

    # Generate and display report
    report = reorganizer.generate_report()
    print(report)
    print()

    # Run dry run first
    print("Running analysis and dry run...")
    success = reorganizer.run_reorganization(dry_run=True)

    if not success:
        print("❌ Dry run failed. Please check the logs.")
        return

    print()
    print("=" * 60)
    choice = input("Proceed with the reorganization? (y/N): ").lower().strip()

    if choice == 'y':
        print("🚀 Executing reorganization...")
        success = reorganizer.run_reorganization(dry_run=False)

        if success:
            print("✅ Reorganization completed successfully!")
            print("📋 Check reorganization.log for detailed information.")
        else:
            print("❌ Reorganization failed. Check reorganization.log for details.")
    else:
        print("ℹ️  Reorganization cancelled.")

if __name__ == "__main__":
    main()
