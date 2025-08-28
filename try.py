def _check_dependencies(self):
        """Check dependency issues"""
        self.logger.info("Analyzing dependencies...")
        
        requirements_file = self.project_path / 'requirements.txt'
        
        if not requirements_file.exists():
            self._add_error(
                file_path=str(requirements_file),
                error_type="MISSING_REQUIREMENTS",
                line_number=0,
                description="Missing requirements.txt file",
                severity="HIGH",
                suggested_fix="Create requirements.txt with project dependencies"
            )
            return
        
        try:
            with open(requirements_file, 'r') as f:
                requirements = f.read()
                
            self.logger.debug(f"Requirements file content: {len(requirements)} characters")
            
            # Check for version pinning
            lines = [line.strip() for line in requirements.strip().split('\n') if line.strip()]
            unpinned_count = 0
            
            for i, line in enumerate(lines, 1):
                if line and not line.startswith('#'):
                    # Check if version is pinned
                    if '==' not in line and '>=' not in line and '<=' not in line and '~=' not in line:
                        unpinned_count += 1
                        self._add_error(
                            file_path=str(requirements_file),
                            error_type="UNPINNED_DEPENDENCY",
                            line_number=i,
                            description=f"Unpinned dependency: {line}",
                            severity="LOW",
                            suggested_fix="Pin dependency versions: package==1.2.3",
                            context=f"Current: {line}"
                        )
            
            # Check for common security issues
            vulnerable_packages = {
                'requests': '2.25.0',  # Example of known vulnerable version
                'urllib3': '1.25.0',
                'jinja2': '2.10.0'
            }
            
            for line in lines:
                for pkg, vulnerable_version in vulnerable_packages.items():
                    if line.startswith(pkg) and vulnerable_version in line:
                        self._add_error(
                            file_path=str(requirements_file),
                            error_type="VULNERABLE_DEPENDENCY",
                            line_number=lines.index(line) + 1,
                            description=f"Potentially vulnerable version of {pkg}",
                            severity="MEDIUM",
                            suggested_fix=f"Update {pkg} to latest stable version"
                        )
            
            self.logger.info(f"Checked {len(lines)} dependencies, {unpinned_count} unpinned")
            
        except Exception as e:
            self.logger.error(f"Error checking requirements.txt: {e}")
            self._add_error(
                file_path=str(requirements_file),
                error_type="REQUIREMENTS_READ_ERROR",
                line_number=0,
                description=f"Could not read requirements.txt: {str(e)}",
                severity="MEDIUM",
                suggested_fix="Fix requirements.txt format and permissions"
            )#!/usr/bin/env python3
"""
Betting Predictor Error Fixer & Validator
A comprehensive tool to detect and fix common issues in betting prediction systems.
"""

import os
import re
import ast
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import importlib.util


@dataclass
class ErrorReport:
    """Data class for error reports"""
    file_path: 
    error_type: str
    line_number: int
    description: str
    severity: str
    suggested_fix: str


    def _add_error(self, file_path: str, error_type: str, line_number: int, 
                   description: str, severity: str, suggested_fix: str, context: str = ""):
        """Add error to list and log it with full details"""
        error = ErrorReport(
            file_path=file_path,
            error_type=error_type,
            line_number=line_number,
            description=description,
            severity=severity,
            suggested_fix=suggested_fix
        )
        
        self.errors.append(error)
        
        # Log the issue with full details
        log_message = (
            f"[{severity}] {error_type} in {file_path}:{line_number} - "
            f"{description} | Fix: {suggested_fix}"
        )
        
        if context:
            log_message += f" | Context: {context}"
            
        # Log to main logger
        if severity == "HIGH":
            self.logger.error(log_message)
        elif severity == "MEDIUM":
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
            
        # Log to issues logger for detailed tracking
        self.issues_logger.info(log_message)
        
        # Also log to console for immediate feedback
        print(f"🔍 Found {severity} issue: {error_type} in {Path(file_path).name}:{line_number}")
        
        return error
    """Main class for fixing betting predictor issues"""
    
    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path)
        self.errors: List[ErrorReport] = []
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        log_dir = self.project_path / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging with detailed formatting
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        
        logging.basicConfig(
            level=logging.DEBUG,
            format=log_format,
            handlers=[
                logging.FileHandler(log_dir / 'betting_fixer.log'),
                logging.FileHandler(log_dir / 'betting_issues.log'),  # Separate issues log
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Create separate logger for issues
        self.issues_logger = logging.getLogger('issues')
        issues_handler = logging.FileHandler(log_dir / 'issues_detailed.log')
        issues_handler.setFormatter(logging.Formatter(
            '%(asctime)s - ISSUE FOUND - %(message)s'
        ))
        self.issues_logger.addHandler(issues_handler)
        self.issues_logger.setLevel(logging.INFO)

    def scan_project(self) -> List[ErrorReport]:
        """Scan entire project for common issues"""
        self.logger.info("="*60)
        self.logger.info("STARTING BETTING PREDICTOR PROJECT SCAN")
        self.logger.info("="*60)
        
        python_files = list(self.project_path.glob("**/*.py"))
        self.logger.info(f"Found {len(python_files)} Python files to scan")
        
        for file_path in python_files:
            self.logger.info(f"Scanning file: {file_path}")
            try:
                self._scan_file(file_path)
                self.logger.debug(f"✅ Successfully scanned {file_path}")
            except Exception as e:
                self.logger.error(f"❌ Error scanning {file_path}: {e}")
                
        self.logger.info("Checking project structure...")
        self._scan_project_structure()
        
        self.logger.info("Checking dependencies...")
        self._check_dependencies()
        
        # Log summary
        self.logger.info("="*60)
        self.logger.info("SCAN COMPLETE")
        self.logger.info(f"Total issues found: {len(self.errors)}")
        
        # Count by severity
        severity_counts = {}
        for error in self.errors:
            severity_counts[error.severity] = severity_counts.get(error.severity, 0) + 1
            
        for severity, count in severity_counts.items():
            self.logger.info(f"{severity} priority issues: {count}")
        self.logger.info("="*60)
        
    def _check_streamlit_issues(self, file_path: Path, content: str):
        """Check for Streamlit-specific issues"""
        self.logger.debug(f"Checking Streamlit issues in {file_path.name}")
        
        if 'streamlit' not in content and 'st.' not in content:
            return
            
        lines = content.split('\n')
        
        # Check for common Streamlit anti-patterns
        streamlit_issues = [
            ('st.cache', 'Deprecated st.cache usage', 'Use @st.cache_data or @st.cache_resource'),
            ('st.beta_', 'Beta function usage', 'Update to stable API'),
            ('st.experimental_', 'Experimental function usage', 'Check if stable version exists')
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, description, fix in streamlit_issues:
                if pattern in line:
                    self._add_error(
                        file_path=str(file_path),
                        error_type="STREAMLIT_DEPRECATED",
                        line_number=i,
                        description=description,
                        severity="MEDIUM",
                        suggested_fix=fix,
                        context=f"Line: {line.strip()}"
                    )

    def _check_ml_pipeline_issues(self, file_path: Path, tree: ast.AST, content: str):
        """Check for ML pipeline issues"""
        self.logger.debug(f"Checking ML pipeline issues in {file_path.name}")
        
        lines = content.split('\n')
        
        # Check for data leakage
        if 'fit_transform' in content and 'test' in content.lower():
            for i, line in enumerate(lines, 1):
                if 'fit_transform' in line and ('test' in line.lower() or 'val' in line.lower()):
                    self._add_error(
                        file_path=str(file_path),
                        error_type="DATA_LEAKAGE",
                        line_number=i,
                        description="Potential data leakage: fit_transform on test data",
                        severity="HIGH",
                        suggested_fix="Use transform() only on test data, fit_transform() on training data",
                        context=f"Line: {line.strip()}"
                    )
        
        # Check for missing train/validation split
        if any(ml_term in content for ml_term in ['fit(', 'train', 'XGB', 'model']):
            if 'train_test_split' not in content and 'cross_val' not in content:
                self._add_error(
                    file_path=str(file_path),
                    error_type="NO_TRAIN_VAL_SPLIT",
                    line_number=1,
                    description="ML training without proper train/validation split",
                    severity="MEDIUM",
                    suggested_fix="Add train_test_split or cross-validation"
                )

    def _check_code_quality(self, file_path: Path, tree: ast.AST, content: str):
        """Check for code quality issues"""
        self.logger.debug(f"Checking code quality in {file_path.name}")
        
        lines = content.split('\n')
        
        # Check for long functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_length = (node.end_lineno or node.lineno) - node.lineno
                if func_length > 50:
                    self._add_error(
                        file_path=str(file_path),
                        error_type="LONG_FUNCTION",
                        line_number=node.lineno,
                        description=f"Function '{node.name}' is {func_length} lines long",
                        severity="LOW",
                        suggested_fix="Consider breaking into smaller functions"
                    )
        
        # Check for missing docstrings
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    self._add_error(
                        file_path=str(file_path),
                        error_type="MISSING_DOCSTRING",
                        line_number=node.lineno,
                        description=f"Missing docstring for {type(node).__name__.lower()}: {node.name}",
                        severity="LOW",
                        suggested_fix="Add docstring explaining purpose and parameters"
                    )

    def _scan_file(self, file_path: Path):
        """Scan individual Python file for issues"""
        self.logger.debug(f"Starting detailed scan of {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            self.logger.debug(f"File content loaded: {len(content)} characters")
                
            # Parse AST for syntax errors
            try:
                tree = ast.parse(content)
                self.logger.debug(f"AST parsing successful for {file_path}")
                
                # Run all checks
                issues_before = len(self.errors)
                
                self._check_imports(file_path, tree, content)
                self._check_exception_handling(file_path, tree, content)
                self._check_api_security(file_path, content)
                self._check_data_validation(file_path, tree, content)
                self._check_thread_safety(file_path, content)
                self._check_model_handling(file_path, content)
                self._check_streamlit_issues(file_path, content)
                self._check_ml_pipeline_issues(file_path, tree, content)
                self._check_code_quality(file_path, tree, content)
                
                issues_found = len(self.errors) - issues_before
                self.logger.info(f"Found {issues_found} issues in {file_path.name}")
                
            except SyntaxError as e:
                error_context = f"Line content: {content.split(chr(10))[e.lineno-1] if e.lineno else 'Unknown'}"
                self._add_error(
                    file_path=str(file_path),
                    error_type="SYNTAX_ERROR",
                    line_number=e.lineno or 0,
                    description=f"Syntax error: {e.msg}",
                    severity="HIGH",
                    suggested_fix="Fix syntax error according to Python grammar rules",
                    context=error_context
                )
                
        except Exception as e:
            self.logger.error(f"Critical error scanning {file_path}: {e}")
            self._add_error(
                file_path=str(file_path),
                error_type="FILE_READ_ERROR",
                line_number=0,
                description=f"Could not read/parse file: {str(e)}",
                severity="HIGH",
                suggested_fix="Check file permissions and encoding"
            )

    def _check_imports(self, file_path: Path, tree: ast.AST, content: str):
        """Check for import-related issues"""
        self.logger.debug(f"Checking imports in {file_path.name}")
        
        imports = []
        import_lines = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
                    import_lines[alias.name] = node.lineno
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
                    import_lines[node.module] = node.lineno
        
        self.logger.debug(f"Found imports: {imports}")
        
        # Check for missing common dependencies
        required_deps = {
            'streamlit': ['st.', 'streamlit'],
            'pandas': ['pd.', 'pandas'],
            'numpy': ['np.', 'numpy'],
            'xgboost': ['xgb.', 'XGBClassifier', 'XGBRegressor'],
            'sklearn': ['sklearn', 'train_test_split'],
            'requests': ['requests.get', 'requests.post'],
            'os': ['os.getenv', 'os.environ'],
            'logging': ['logging.', 'logger']
        }
        
        for dep, patterns in required_deps.items():
            if any(pattern in content for pattern in patterns):
                if dep not in imports and not any(dep in imp for imp in imports):
                    context = f"Usage patterns found: {[p for p in patterns if p in content]}"
                    self._add_error(
                        file_path=str(file_path),
                        error_type="MISSING_IMPORT",
                        line_number=1,
                        description=f"Missing import for {dep}",
                        severity="HIGH",
                        suggested_fix=f"Add: import {dep}",
                        context=context
                    )
        
        # Check for unused imports
        for imp in imports:
            if imp not in content.replace(f"import {imp}", ""):
                self._add_error(
                    file_path=str(file_path),
                    error_type="UNUSED_IMPORT",
                    line_number=import_lines.get(imp, 1),
                    description=f"Unused import: {imp}",
                    severity="LOW",
                    suggested_fix=f"Remove unused import: {imp}"
                )

    def _check_exception_handling(self, file_path: Path, tree: ast.AST, content: str):
        """Check for proper exception handling"""
        self.logger.debug(f"Checking exception handling in {file_path.name}")
        
        api_calls = []
        try_blocks = []
        file_operations = []
        
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check for API calls
                if hasattr(node.func, 'attr'):
                    if node.func.attr in ['get', 'post', 'request', 'put', 'delete']:
                        line_content = lines[node.lineno-1] if node.lineno-1 < len(lines) else ""
                        api_calls.append((node.lineno, line_content.strip()))
                        
                # Check for file operations
                if hasattr(node.func, 'id'):
                    if node.func.id in ['open', 'load', 'dump']:
                        line_content = lines[node.lineno-1] if node.lineno-1 < len(lines) else ""
                        file_operations.append((node.lineno, line_content.strip()))
                        
            elif isinstance(node, ast.Try):
                try_blocks.extend(range(node.lineno, node.end_lineno or node.lineno + 1))
        
        self.logger.debug(f"Found {len(api_calls)} API calls, {len(file_operations)} file operations")
        
        # Check if API calls are wrapped in try-catch
        for api_line, api_code in api_calls:
            if api_line not in try_blocks:
                self._add_error(
                    file_path=str(file_path),
                    error_type="MISSING_EXCEPTION_HANDLING",
                    line_number=api_line,
                    description="API call without exception handling",
                    severity="MEDIUM",
                    suggested_fix="Wrap API calls in try-except blocks",
                    context=f"Code: {api_code}"
                )
        
        # Check file operations
        for file_line, file_code in file_operations:
            if file_line not in try_blocks:
                self._add_error(
                    file_path=str(file_path),
                    error_type="MISSING_FILE_EXCEPTION_HANDLING",
                    line_number=file_line,
                    description="File operation without exception handling",
                    severity="MEDIUM",
                    suggested_fix="Wrap file operations in try-except blocks",
                    context=f"Code: {file_code}"
                )

    def _check_api_security(self, file_path: Path, content: str):
        """Check for API security issues"""
        self.logger.debug(f"Checking API security in {file_path.name}")
        
        lines = content.split('\n')
        
        # Enhanced API key patterns
        api_key_patterns = [
            (r'api_key\s*=\s*["\'][a-zA-Z0-9_-]{10,}["\']', "API key assignment"),
            (r'API_KEY\s*=\s*["\'][a-zA-Z0-9_-]{10,}["\']', "API key constant"),
            (r'token\s*=\s*["\'][a-zA-Z0-9_-]{20,}["\']', "Token assignment"),
            (r'password\s*=\s*["\'][^"\']{3,}["\']', "Password hardcoded"),
            (r'secret\s*=\s*["\'][a-zA-Z0-9_-]{10,}["\']', "Secret key"),
            (r'key\s*=\s*["\'][a-zA-Z0-9_-]{15,}["\']', "Generic key")
        ]
        
        security_issues_found = 0
        
        for i, line in enumerate(lines, 1):
            for pattern, description in api_key_patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    security_issues_found += 1
                    self._add_error(
                        file_path=str(file_path),
                        error_type="HARDCODED_CREDENTIALS",
                        line_number=i,
                        description=f"Hardcoded credentials detected: {description}",
                        severity="HIGH",
                        suggested_fix="Use environment variables: os.getenv('API_KEY')",
                        context=f"Matched text: {match.group()[:20]}..."
                    )
        
        # Check for missing environment variable usage
        if 'getenv' not in content and 'environ' not in content and security_issues_found == 0:
            if any(word in content.lower() for word in ['api', 'key', 'token', 'password']):
                self._add_error(
                    file_path=str(file_path),
                    error_type="NO_ENV_VAR_USAGE",
                    line_number=1,
                    description="No environment variable usage found for sensitive data",
                    severity="MEDIUM",
                    suggested_fix="Import os and use os.getenv() for sensitive configuration"
                )
        
        self.logger.debug(f"Found {security_issues_found} security issues in {file_path.name}")

    def _check_data_validation(self, file_path: Path, tree: ast.AST, content: str):
        """Check for data validation issues"""
        # Look for pandas operations without validation
        if 'pandas' in content or 'pd.' in content:
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if '.read_csv' in line or '.read_json' in line:
                    # Check if there's validation after data loading
                    next_lines = lines[i:i+10]  # Check next 10 lines
                    if not any('isna()' in l or 'isnull()' in l or 'dropna()' in l 
                              for l in next_lines):
                        self.errors.append(ErrorReport(
                            file_path=str(file_path),
                            error_type="MISSING_DATA_VALIDATION",
                            line_number=i,
                            description="Data loading without validation",
                            severity="MEDIUM",
                            suggested_fix="Add data validation: df.isna().sum(), df.dropna()"
                        ))

    def _check_thread_safety(self, file_path: Path, content: str):
        """Check for thread safety issues"""
        if 'threading' in content or 'concurrent' in content:
            if 'lock' not in content.lower() and 'mutex' not in content.lower():
                self.errors.append(ErrorReport(
                    file_path=str(file_path),
                    error_type="THREAD_SAFETY",
                    line_number=1,
                    description="Threading without synchronization mechanisms",
                    severity="MEDIUM",
                    suggested_fix="Add threading.Lock() for shared resources"
                ))

    def _check_model_handling(self, file_path: Path, content: str):
        """Check for ML model handling issues"""
        if 'pickle' in content or 'joblib' in content:
            if 'load' in content and 'FileNotFoundError' not in content:
                self.errors.append(ErrorReport(
                    file_path=str(file_path),
                    error_type="MODEL_LOADING_ERROR",
                    line_number=1,
                    description="Model loading without file existence check",
                    severity="MEDIUM",
                    suggested_fix="Check if model file exists before loading"
                ))

    def _scan_project_structure(self):
        """Check project structure issues"""
        self.logger.info("Checking project structure...")
        
        required_files = {
            'requirements.txt': 'Python dependencies',
            '.env.example': 'Environment variables template',
            'README.md': 'Project documentation',
            '.gitignore': 'Git ignore rules'
        }
        
        optional_files = {
            'config.py': 'Configuration management',
            'utils.py': 'Utility functions',
            'tests/': 'Test directory'
        }
        
        # Check required files
        for required_file, description in required_files.items():
            file_path = self.project_path / required_file
            if not file_path.exists():
                self._add_error(
                    file_path=str(self.project_path),
                    error_type="MISSING_REQUIRED_FILE",
                    line_number=0,
                    description=f"Missing required file: {required_file} ({description})",
                    severity="MEDIUM",
                    suggested_fix=f"Create {required_file}",
                    context=f"File purpose: {description}"
                )
            else:
                self.logger.debug(f"✅ Found required file: {required_file}")
        
        # Check optional but recommended files
        for optional_file, description in optional_files.items():
            file_path = self.project_path / optional_file
            if not file_path.exists():
                self._add_error(
                    file_path=str(self.project_path),
                    error_type="MISSING_RECOMMENDED_FILE",
                    line_number=0,
                    description=f"Missing recommended file: {optional_file} ({description})",
                    severity="LOW",
                    suggested_fix=f"Consider creating {optional_file}",
                    context=f"Purpose: {description}"
                )
        
        # Check directory structure
        recommended_dirs = ['logs', 'data', 'models', 'tests']
        for dir_name in recommended_dirs:
            dir_path = self.project_path / dir_name
            if not dir_path.exists():
                self.logger.debug(f"Missing recommended directory: {dir_name}")
        
        self.logger.info("Project structure check complete")

    def _check_dependencies(self):
        """Check dependency issues"""
        requirements_file = self.project_path / 'requirements.txt'
        
        if requirements_file.exists():
            try:
                with open(requirements_file, 'r') as f:
                    requirements = f.read()
                    
                # Check for version pinning
                lines = requirements.strip().split('\n')
                for line in lines:
                    if line.strip() and '==' not in line and '>=' not in line:
                        self.errors.append(ErrorReport(
                            file_path=str(requirements_file),
                            error_type="UNPINNED_DEPENDENCY",
                            line_number=0,
                            description=f"Unpinned dependency: {line}",
                            severity="LOW",
                            suggested_fix="Pin dependency versions: package==1.2.3"
                        ))
            except Exception as e:
                self.logger.error(f"Error checking requirements.txt: {e}")

    def generate_fixes(self) -> Dict[str, str]:
        """Generate automated fixes for common issues"""
        fixes = {}
        
        # Generate requirements.txt
        fixes['requirements.txt'] = self._generate_requirements()
        
        # Generate .env.example
        fixes['.env.example'] = self._generate_env_example()
        
        # Generate error handling wrapper
        fixes['error_handler.py'] = self._generate_error_handler()
        
        # Generate data validator
        fixes['data_validator.py'] = self._generate_data_validator()
        
        # Generate API client with proper error handling
        fixes['api_client.py'] = self._generate_api_client()
        
        return fixes

    def _generate_requirements(self) -> str:
        """Generate requirements.txt with common dependencies"""
        return """streamlit==1.28.1
pandas==2.0.3
numpy==1.24.3
xgboost==1.7.6
scikit-learn==1.3.0
requests==2.31.0
python-dotenv==1.0.0
joblib==1.3.2
plotly==5.17.0
seaborn==0.12.2
matplotlib==3.7.2
"""

    def _generate_env_example(self) -> str:
        """Generate .env.example file"""
        return """# API Keys (Replace with your actual keys)
SPORTS_API_KEY=your_sports_api_key_here
BETTING_ODDS_API_KEY=your_betting_odds_api_key_here

# Database Configuration
DATABASE_URL=sqlite:///betting_data.db

# Application Settings
DEBUG=False
LOG_LEVEL=INFO

# Rate Limiting
API_RATE_LIMIT=60
CACHE_TIMEOUT=300
"""

    def _generate_error_handler(self) -> str:
        """Generate error handling utility"""
        return """import logging
import functools
from typing import Any, Callable, Optional
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class ErrorHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def with_retry(max_retries: int = 3, backoff_factor: float = 0.3):
        \"\"\"Decorator for API calls with retry logic\"\"\"
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise e
                        time.sleep(backoff_factor * (2 ** attempt))
                return None
            return wrapper
        return decorator
    
    @staticmethod
    def safe_api_call(url: str, headers: Optional[dict] = None, timeout: int = 30) -> Optional[dict]:
        \"\"\"Safe API call with proper error handling\"\"\"
        try:
            session = requests.Session()
            retry = Retry(
                total=3,
                backoff_factor=0.3,
                status_forcelist=[500, 502, 503, 504]
            )
            adapter = HTTPAdapter(max_retries=retry)
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            
            response = session.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"API call failed: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return None
"""

    def _generate_data_validator(self) -> str:
        """Generate data validation utility"""
        return """import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

class DataValidator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_dataframe(self, df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
        \"\"\"Validate DataFrame structure and content\"\"\"
        errors = []
        
        # Check if DataFrame is empty
        if df.empty:
            errors.append("DataFrame is empty")
            return False, errors
        
        # Check required columns
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        
        # Check for null values in critical columns
        null_counts = df[required_columns].isnull().sum()
        critical_nulls = null_counts[null_counts > 0]
        if not critical_nulls.empty:
            errors.append(f"Null values in critical columns: {dict(critical_nulls)}")
        
        # Check for duplicates
        if df.duplicated().any():
            errors.append(f"Found {df.duplicated().sum()} duplicate rows")
        
        return len(errors) == 0, errors
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Clean and preprocess data\"\"\"
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna('Unknown')
        
        return df
    
    def validate_betting_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        \"\"\"Specific validation for betting data\"\"\"
        errors = []
        
        # Check for reasonable odds values
        if 'odds' in df.columns:
            invalid_odds = df[(df['odds'] <= 0) | (df['odds'] > 100)]
            if not invalid_odds.empty:
                errors.append(f"Invalid odds values: {len(invalid_odds)} rows")
        
        # Check date formats
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            try:
                pd.to_datetime(df[col])
            except:
                errors.append(f"Invalid date format in column: {col}")
        
        return len(errors) == 0, errors
"""

    def _generate_api_client(self) -> str:
        """Generate API client with proper error handling"""
        return """import os
import time
import requests
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging

@dataclass
class APIConfig:
    base_url: str
    api_key: str
    rate_limit: int = 60  # requests per minute
    timeout: int = 30

class RateLimiter:
    def __init__(self, max_calls: int, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    def wait_if_needed(self):
        now = time.time()
        # Remove old calls outside time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        if len(self.calls) >= self.max_calls:
            sleep_time = self.time_window - (now - self.calls[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.calls.append(now)

class BettingAPIClient:
    def __init__(self, config: APIConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config.rate_limit)
        self.logger = logging.getLogger(__name__)
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({
            'Authorization': f'Bearer {self.config.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'BettingPredictor/1.0'
        })
        return session
    
    def get_data(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict]:
        \"\"\"Get data from API with rate limiting and error handling\"\"\"
        self.rate_limiter.wait_if_needed()
        
        url = f"{self.config.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:  # Rate limit exceeded
                self.logger.warning("Rate limit exceeded, waiting...")
                time.sleep(60)
                return self.get_data(endpoint, params)  # Retry
            else:
                self.logger.error(f"HTTP error {response.status_code}: {e}")
                return None
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            return None
        
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return None
    
    @classmethod
    def from_env(cls) -> 'BettingAPIClient':
        \"\"\"Create API client from environment variables\"\"\"
        config = APIConfig(
            base_url=os.getenv('SPORTS_API_BASE_URL', ''),
            api_key=os.getenv('SPORTS_API_KEY', ''),
            rate_limit=int(os.getenv('API_RATE_LIMIT', '60')),
            timeout=int(os.getenv('API_TIMEOUT', '30'))
        )
        return cls(config)
"""

    def apply_fixes(self, fixes: Dict[str, str]):
        """Apply generated fixes to the project"""
        self.logger.info("="*50)
        self.logger.info("APPLYING AUTOMATIC FIXES")
        self.logger.info("="*50)
        
        fixes_applied = 0
        fixes_failed = 0
        
        for filename, content in fixes.items():
            file_path = self.project_path / filename
            
            try:
                # Create directory if it doesn't exist
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Backup existing file if it exists
                if file_path.exists():
                    backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
                    file_path.rename(backup_path)
                    self.logger.info(f"📁 Backed up existing {filename} to {backup_path.name}")
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.logger.info(f"✅ Created/updated {filename}")
                fixes_applied += 1
                
            except Exception as e:
                self.logger.error(f"❌ Failed to create {filename}: {e}")
                fixes_failed += 1
        
        self.logger.info(f"Fixes applied: {fixes_applied}, Failed: {fixes_failed}")
        self.logger.info("="*50)

    def generate_report(self) -> str:
        """Generate comprehensive error report"""
        report = []
        report.append("=" * 80)
        report.append("🔍 BETTING PREDICTOR COMPREHENSIVE ERROR REPORT")
        report.append("=" * 80)
        
        # Summary statistics
        total_issues = len(self.errors)
        high_errors = [e for e in self.errors if e.severity == "HIGH"]
        medium_errors = [e for e in self.errors if e.severity == "MEDIUM"]
        low_errors = [e for e in self.errors if e.severity == "LOW"]
        
        report.append(f"\n📊 SUMMARY:")
        report.append(f"   Total Issues Found: {total_issues}")
        report.append(f"   🔴 High Priority: {len(high_errors)}")
        report.append(f"   🟡 Medium Priority: {len(medium_errors)}")
        report.append(f"   🟢 Low Priority: {len(low_errors)}")
        
        # Issues by type
        error_types = {}
        for error in self.errors:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
        
        report.append(f"\n📋 ISSUES BY TYPE:")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            report.append(f"   {error_type}: {count}")
        
        # Issues by file
        file_issues = {}
        for error in self.errors:
            file_name = Path(error.file_path).name
            file_issues[file_name] = file_issues.get(file_name, 0) + 1
        
        report.append(f"\n📁 ISSUES BY FILE:")
        for file_name, count in sorted(file_issues.items(), key=lambda x: x[1], reverse=True):
            report.append(f"   {file_name}: {count} issues")
        
        # Detailed issues by severity
        for severity, errors, emoji in [("HIGH", high_errors, "🔴"), ("MEDIUM", medium_errors, "🟡"), ("LOW", low_errors, "🟢")]:
            if errors:
                report.append(f"\n{emoji} {severity} PRIORITY ISSUES ({len(errors)} total)")
                report.append("=" * 60)
                
                for i, error in enumerate(errors, 1):
                    report.append(f"\n{i}. {error.error_type}")
                    report.append(f"   📁 File: {Path(error.file_path).name}")
                    report.append(f"   📍 Line: {error.line_number}")
                    report.append(f"   📝 Description: {error.description}")
                    report.append(f"   🔧 Suggested Fix: {error.suggested_fix}")
                    
                    # Add context if available
                    if hasattr(error, 'context') and error.context:
                        report.append(f"   💡 Context: {error.context}")
        
        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS:")
        if high_errors:
            report.append("   1. 🚨 Address HIGH priority issues immediately")
            report.append("      These can cause runtime errors or security vulnerabilities")
        
        if medium_errors:
            report.append("   2. ⚠️  Fix MEDIUM priority issues before production")
            report.append("      These affect reliability and maintainability")
        
        if low_errors:
            report.append("   3. 📈 Consider LOW priority improvements")
            report.append("      These enhance code quality and readability")
        
        report.append(f"\n🛠️  To apply automatic fixes, run with --fix flag")
        report.append(f"📊 Report generated at: {Path().absolute()}")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main function to run the fixer"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive Betting Predictor Issue Fixer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python betting_fixer.py --path . --scan                    # Scan current directory
  python betting_fixer.py --path /project --report --fix     # Full scan and fix
  python betting_fixer.py --verbose --fix                    # Verbose logging with fixes
        """
    )
    
    parser.add_argument("--path", default=".", help="Project path to scan (default: current directory)")
    parser.add_argument("--scan", action="store_true", help="Scan for issues (default action)")
    parser.add_argument("--fix", action="store_true", help="Apply automatic fixes")
    parser.add_argument("--report", action="store_true", help="Generate detailed error report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--output", "-o", help="Output file for report (default: error_report.txt)")
    
    args = parser.parse_args()
    
    # If no action specified, default to scan and report
    if not any([args.scan, args.fix, args.report]):
        args.scan = True
        args.report = True
    
    print("🔍 Betting Predictor Error Fixer v1.0")
    print("=" * 50)
    
    try:
        fixer = BettingPredictorFixer(args.path)
        
        # Adjust logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Scan for errors
        if args.scan or args.fix or args.report:
            print(f"🔎 Scanning project at: {Path(args.path).absolute()}")
            fixer.scan_project()
        
        # Generate and display report
        if args.report:
            report = fixer.generate_report()
            print("\n" + report)
            
            # Save to file
            output_file = args.output or "error_report.txt"
            with open(output_file, "w") as f:
                f.write(report)
            print(f"\n📄 Detailed report saved to: {output_file}")
        
        # Apply fixes
        if args.fix:
            print("\n🛠️  Generating and applying fixes...")
            fixes = fixer.generate_fixes()
            fixer.apply_fixes(fixes)
            print("✅ Automatic fixes applied!")
        
        # Final summary
        total_issues = len(fixer.errors)
        high_issues = len([e for e in fixer.errors if e.severity == "HIGH"])
        
        print(f"\n📊 FINAL SUMMARY:")
        print(f"   Total issues found: {total_issues}")
        print(f"   High priority issues: {high_issues}")
        
        if high_issues > 0:
            print("   ⚠️  Please address high priority issues before deployment!")
        elif total_issues == 0:
            print("   🎉 No issues found! Your code looks good!")
        else:
            print("   👍 No critical issues found!")
            
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        logging.exception("Unexpected error in main()")
    
    return len(fixer.errors) if 'fixer' in locals() else 1


if __name__ == "__main__":
    main()