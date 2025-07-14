#!/usr/bin/env python3
"""
Dynamic Project Checklist Generator
Automatically checks project status based on actual files and code.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any

class ProjectChecker:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.status = {}
    
    def file_exists(self, path: str) -> bool:
        """Check if a file exists."""
        return (self.project_root / path).exists()
    
    def directory_exists(self, path: str) -> bool:
        """Check if a directory exists."""
        return (self.project_root / path).is_dir()
    
    def file_contains(self, path: str, content: str) -> bool:
        """Check if a file contains specific content."""
        try:
            file_path = self.project_root / path
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return content in f.read()
        except Exception:
            pass
        return False
    
    def count_files_in_dir(self, path: str, pattern: str = "*") -> int:
        """Count files matching pattern in directory."""
        try:
            dir_path = self.project_root / path
            if dir_path.is_dir():
                return len(list(dir_path.glob(pattern)))
        except Exception:
            pass
        return 0
    
    def has_git_repo(self) -> bool:
        """Check if project is a git repository."""
        return (self.project_root / ".git").exists()
    
    def check_phase_1(self) -> Dict[str, Tuple[bool, str]]:
        """Phase 1: Planning & Setup"""
        return {
            "Architecture docs": (
                self.directory_exists("docs") and self.count_files_in_dir("docs", "*.md") > 0,
                "docs/ directory with markdown files"
            ),
            "Version control": (
                self.has_git_repo(),
                ".git directory found"
            ),
            "CI/CD pipeline": (
                self.file_exists(".github/workflows/ci-cd.yml") or 
                self.count_files_in_dir(".github/workflows", "*.yml") > 0,
                ".github/workflows/ with YAML files"
            ),
            "Cloud setup": (
                self.file_exists("cloud-setup-example.tf") or 
                self.count_files_in_dir(".", "*.tf") > 0 or
                self.file_contains(".env.example", "AWS") or
                self.file_contains(".env.example", "GCP") or
                self.file_contains(".env.example", "AZURE"),
                "Terraform files or cloud config in .env.example"
            )
        }
    
    def check_phase_2(self) -> Dict[str, Tuple[bool, str]]:
        """Phase 2: Core ML Pipeline Automation"""
        return {
            "ML pipeline code": (
                self.directory_exists("ml_pipeline") and 
                self.file_exists("ml_pipeline/pipeline.py"),
                "ml_pipeline/ directory with pipeline.py"
            ),
            "Dockerfiles": (
                self.file_exists("Dockerfile") or 
                self.count_files_in_dir(".", "*Dockerfile*") > 0,
                "Dockerfile found in project"
            ),
            "Kubernetes manifests": (
                self.directory_exists("k8s") and 
                self.count_files_in_dir("k8s", "*.yml") > 0,
                "k8s/ directory with YAML manifests"
            ),
            "Monitoring config": (
                self.directory_exists("monitoring") and 
                self.file_exists("monitoring/prometheus.yml"),
                "monitoring/ with Prometheus config"
            ),
            "Tests": (
                self.directory_exists("tests") and 
                self.count_files_in_dir("tests", "test_*.py") > 0,
                "tests/ directory with test files"
            ),
            "Grafana dashboards": (
                self.file_contains("monitoring", "grafana") or
                self.count_files_in_dir("monitoring", "*grafana*.json") > 0,
                "Grafana dashboard files in monitoring/"
            )
        }
    
    def check_phase_3(self) -> Dict[str, Tuple[bool, str]]:
        """Phase 3: Distributed Systems Simulation"""
        return {
            "Raft consensus module": (
                self.file_exists("distributed_sim/raft.py") and
                self.file_contains("distributed_sim/raft.py", "class RaftNode"),
                "distributed_sim/raft.py with RaftNode class"
            ),
            "React UI for Raft": (
                self.directory_exists("frontend") and
                self.file_exists("frontend/src/App.js") and
                self.file_contains("frontend/src/App.js", "raft"),
                "frontend/src/App.js with Raft UI components"
            ),
            "Metrics/tracing": (
                self.file_exists("monitoring/telemetry.py"),
                "monitoring/telemetry.py exists"
            )
        }
    
    def check_phase_4(self) -> Dict[str, Tuple[bool, str]]:
        """Phase 4: AI Chatbot Interface"""
        return {
            "Chatbot backend": (
                self.directory_exists("chatbot") and 
                self.file_exists("chatbot/bot.py"),
                "chatbot/ directory with bot.py"
            ),
            "API integration": (
                self.file_exists("chatbot/api.py") and
                self.file_contains("chatbot/bot.py", "ml_pipeline"),
                "chatbot integration with ML pipeline APIs"
            ),
            "Vector DB integration": (
                self.file_contains("chatbot/bot.py", "faiss") or
                self.file_contains("chatbot/api.py", "faiss") or
                self.file_contains("requirements.txt", "faiss"),
                "FAISS vector database integration"
            ),
            "Slack bot": (
                self.file_exists("chatbot/slack_bot.py") or
                self.file_contains("requirements.txt", "slack"),
                "Slack bot implementation or dependencies"
            )
        }
    
    def check_phase_5(self) -> Dict[str, Tuple[bool, str]]:
        """Phase 5: Telemetry & Production Hardening"""
        return {
            "OpenTelemetry": (
                self.file_exists("monitoring/telemetry.py") and
                self.file_contains("monitoring/telemetry.py", "opentelemetry"),
                "OpenTelemetry instrumentation in telemetry.py"
            ),
            "Prometheus alerts": (
                self.file_exists("monitoring/alert_rules.yml") or
                self.count_files_in_dir("monitoring", "*alert*.yml") > 0,
                "Prometheus alert rules in monitoring/"
            ),
            "HPA configs": (
                self.count_files_in_dir("k8s", "*hpa*.yml") > 0,
                "Horizontal Pod Autoscaler configs in k8s/"
            ),
            "Auth/RBAC": (
                self.file_exists("k8s/rbac.yml") or
                self.count_files_in_dir("k8s", "*rbac*.yml") > 0,
                "RBAC configurations in k8s/"
            ),
            "End-to-end tests": (
                self.file_exists("tests/test_e2e.py") or
                self.count_files_in_dir("tests", "*e2e*.py") > 0,
                "End-to-end test files in tests/"
            )
        }
    
    def check_phase_6(self) -> Dict[str, Tuple[bool, str]]:
        """Phase 6: Documentation, Demo & Presentation"""
        return {
            "README": (
                self.file_exists("README.md"),
                "README.md in project root"
            ),
            "SOPs/docs": (
                self.directory_exists("docs") and
                self.count_files_in_dir("docs", "*.md") > 1,
                "Multiple documentation files in docs/"
            ),
            "Demo script": (
                self.file_exists("docs/demo_script.md") or
                self.count_files_in_dir("docs", "*demo*.md") > 0,
                "Demo script in docs/"
            ),
            "Walkthrough videos": (
                self.count_files_in_dir("docs", "*.mp4") > 0 or
                self.count_files_in_dir(".", "*.mp4") > 0 or
                self.file_contains("docs", "video") or
                self.file_contains("README.md", "video"),
                "Video files or video references found"
            )
        }
    
    def generate_checklist(self) -> str:
        """Generate the complete dynamic checklist."""
        phases = [
            ("Phase 1: Planning & Setup", self.check_phase_1()),
            ("Phase 2: Core ML Pipeline Automation", self.check_phase_2()),
            ("Phase 3: Distributed Systems Simulation", self.check_phase_3()),
            ("Phase 4: AI Chatbot Interface", self.check_phase_4()),
            ("Phase 5: Telemetry & Production Hardening", self.check_phase_5()),
            ("Phase 6: Documentation, Demo & Presentation", self.check_phase_6())
        ]
        
        checklist = "# Dynamic Project Completion Checklist\n\n"
        checklist += f"*Generated automatically on {self.get_timestamp()}*\n\n"
        
        total_items = 0
        completed_items = 0
        
        for phase_name, phase_checks in phases:
            checklist += f"## {phase_name}\n"
            
            for item_name, (is_complete, description) in phase_checks.items():
                status = "[x]" if is_complete else "[ ]"
                checklist += f"- {status} **{item_name}** ({description})\n"
                
                total_items += 1
                if is_complete:
                    completed_items += 1
            
            checklist += "\n"
        
        # Add summary
        completion_rate = (completed_items / total_items * 100) if total_items > 0 else 0
        checklist += "---\n\n"
        checklist += f"## Summary\n"
        checklist += f"- **Total Items:** {total_items}\n"
        checklist += f"- **Completed:** {completed_items}\n"
        checklist += f"- **Completion Rate:** {completion_rate:.1f}%\n\n"
        checklist += "Legend: [x] = Complete, [ ] = Missing/Incomplete\n\n"
        checklist += "*This checklist is generated dynamically by analyzing your project files.*\n"
        
        return checklist
    
    def get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def save_checklist(self, filename: str = "CHECKLIST.md"):
        """Save the dynamic checklist to a file."""
        checklist_content = self.generate_checklist()
        with open(self.project_root / filename, 'w', encoding='utf-8') as f:
            f.write(checklist_content)
        print(f"Dynamic checklist saved to {filename}")
        return checklist_content

if __name__ == "__main__":
    checker = ProjectChecker()
    checklist = checker.save_checklist()
    print("\n" + "="*50)
    print("DYNAMIC PROJECT CHECKLIST")
    print("="*50)
    print(checklist)
