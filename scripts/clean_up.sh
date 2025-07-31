#!/usr/bin/env python3
"""
File Organization Script for mars-gis Repository
Moves files from root directory to appropriate subfolders and Docker directories
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('file_organization.log'),
        logging.StreamHandler()
    ]
)

class MarsGISFileOrganizer:
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.dry_run = False

        # Define file organization rules
        self.organization_rules = {
            # Docker-related files
            "docker/": {
                "patterns": [
                    "Dockerfile*",
                    "docker-compose*.yml",
                    "*docker*.sh",
                    "docker_*.md"
                ],
                "description": "Docker configuration and compose files"
            },

            # Scripts directory
            "scripts/": {
                "patterns": [
                    "*.sh",
                    "cleanup-*.sh",
                    "run_*.sh",
                    "migrate_*.sh",
                    "validate_*.sh",
                    "final_*.sh",
                    "quick_*.sh"
                ],
                "description": "Shell scripts and automation tools"
            },

            # Configuration files
            "config/": {
                "patterns": [
                    ".env*",
                    "*.ini",
                    "*.toml",
                    "cypress.config.*",
                    ".editorconfig"
                ],
                "description": "Configuration files"
            },

            # Documentation
            "docs/": {
                "patterns": [
                    "*.md",
                    "CHANGELOG.md",
                    "TDD_COMPLETION_STATUS.md",
                    "DOCKER_MIGRATION_SUMMARY.md"
                ],
                "description": "Documentation and markdown files"
            },

            # Testing
            "tests/": {
                "patterns": [
                    "pytest.ini",
                    "*test*.py"
                ],
                "description": "Testing configuration and test files"
            }
        }

        # Files to exclude from moving (keep in root)
        self.exclude_files = {
            "README.md",
            "LICENSE",
            ".gitignore",
            "requirements.txt",
            "pyproject.toml"
        }

    def get_matching_files(self) -> Dict[str, List[Path]]:
        """Get files that match organization patterns"""
        matches = {folder: [] for folder in self.organization_rules.keys()}

        # Get all files in root directory
        root_files = [f for f in self.root_dir.iterdir() if f.is_file()]

        for file_path in root_files:
            # Skip excluded files
            if file_path.name in self.exclude_files:
                continue

            # Check against each organization rule
            for target_folder, rules in self.organization_rules.items():
                for pattern in rules["patterns"]:
                    if file_path.match(pattern):
                        matches[target_folder].append(file_path)
                        break

        return matches

    def create_directories(self, directories: List[str]) -> None:
        """Create target directories if they don't exist"""
        for directory in directories:
            dir_path = self.root_dir / directory
            if not dir_path.exists():
                if not self.dry_run:
                    dir_path.mkdir(parents=True, exist_ok=True)
                logging.info(f"Created directory: {dir_path}")

    def move_file(self, source: Path, target_dir: str) -> bool:
        """Move a single file to target directory"""
        target_path = self.root_dir / target_dir / source.name

        try:
            if target_path.exists():
                logging.warning(f"Target file already exists: {target_path}")
                return False

            if not self.dry_run:
                shutil.move(str(source), str(target_path))

            logging.info(f"Moved: {source.name} -> {target_dir}")
            return True

        except Exception as e:
            logging.error(f"Error moving {source.name}: {str(e)}")
            return False

    def organize_files(self, dry_run: bool = True) -> Tuple[int, int]:
        """Organize files according to rules"""
        self.dry_run = dry_run

        if dry_run:
            logging.info("DRY RUN MODE - No files will be moved")

        matches = self.get_matching_files()

        # Create directories
        directories_to_create = [folder for folder, files in matches.items() if files]
        self.create_directories(directories_to_create)

        moved_count = 0
        total_files = sum(len(files) for files in matches.values())

        # Move files
        for target_folder, files in matches.items():
            if not files:
                continue

            logging.info(f"\nMoving files to {target_folder}:")
            logging.info(f"Description: {self.organization_rules[target_folder]['description']}")

            for file_path in files:
                if self.move_file(file_path, target_folder):
                    moved_count += 1

        return moved_count, total_files

    def generate_report(self) -> str:
        """Generate organization report"""
        matches = self.get_matching_files()

        report = []
        report.append("MARS-GIS File Organization Report")
        report.append("=" * 50)

        for target_folder, files in matches.items():
            if files:
                report.append(f"\n{target_folder.upper()}")
                report.append(f"Description: {self.organization_rules[target_folder]['description']}")
                report.append(f"Files to move ({len(files)}):")
                for file_path in files:
                    report.append(f"  - {file_path.name}")

        # Files that will remain in root
        root_files = [f for f in self.root_dir.iterdir() if f.is_file()]
        remaining_files = []

        for file_path in root_files:
            if file_path.name in self.exclude_files:
                remaining_files.append(file_path.name)
            else:
                # Check if file matches any pattern
                matches_pattern = False
                for rules in self.organization_rules.values():
                    for pattern in rules["patterns"]:
                        if file_path.match(pattern):
                            matches_pattern = True
                            break
                    if matches_pattern:
                        break

                if not matches_pattern:
                    remaining_files.append(file_path.name)

        if remaining_files:
            report.append("\nFILES REMAINING IN ROOT:")
            for filename in remaining_files:
                report.append(f"  - {filename}")

        return "\n".join(report)

def main():
    """Main function to run file organization"""
    organizer = MarsGISFileOrganizer()

    print("MARS-GIS File Organization Script")
    print("=" * 40)

    # Generate and display report
    report = organizer.generate_report()
    print(report)

    # Ask user for confirmation
    print("\n" + "=" * 40)
    choice = input("Do you want to proceed with file organization? (y/N): ").lower().strip()

    if choice == 'y':
        # First run in dry-run mode
        print("\nRunning dry-run simulation...")
        moved_count, total_files = organizer.organize_files(dry_run=True)

        print(f"\nDry-run complete. Would move {moved_count}/{total_files} files.")

        # Confirm actual execution
        final_choice = input("Execute the actual file moves? (y/N): ").lower().strip()

        if final_choice == 'y':
            print("\nExecuting file organization...")
            moved_count, total_files = organizer.organize_files(dry_run=False)
            print(f"\nFile organization complete! Moved {moved_count}/{total_files} files.")
            print("Check file_organization.log for detailed logs.")
        else:
            print("Operation cancelled.")
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    main()
