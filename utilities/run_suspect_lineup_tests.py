#!/usr/bin/env python3
"""
Test runner for Suspect Lineup feature tests.

This script runs the comprehensive test suite for the suspect lineup feature,
including unit tests, integration tests, and GUI tests.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_tests():
    """Run the suspect lineup test suite."""
    print("="*80)
    print("SUSPECT LINEUP TEST SUITE")
    print("="*80)
    
    # Ensure we're in the correct directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Test commands to run
    test_commands = [
        {
            "name": "Database Function Tests",
            "cmd": ["python", "-m", "pytest", "tests/test_suspect_lineup_db.py", 
                   "-v", "--tb=short", "-m", "not integration"],
            "description": "Testing database functions with fallback logic"
        },
        {
            "name": "GUI Component Tests", 
            "cmd": ["python", "-m", "pytest", "tests/test_suspect_lineup_gui.py",
                   "-v", "--tb=short", "-m", "not integration"],
            "description": "Testing GUI components and user interactions"
        },
        {
            "name": "Integration Tests",
            "cmd": ["python", "-m", "pytest", "tests/test_suspect_lineup_integration.py",
                   "-v", "--tb=short", "-m", "integration"],
            "description": "Testing end-to-end workflows and component integration"
        },
        {
            "name": "All Suspect Lineup Tests",
            "cmd": ["python", "-m", "pytest", 
                   "tests/test_suspect_lineup_db.py",
                   "tests/test_suspect_lineup_gui.py", 
                   "tests/test_suspect_lineup_integration.py",
                   "-v", "--tb=short", "--cov=suspect_lineup", "--cov=sync_database",
                   "--cov-report=term-missing", "--cov-report=html:htmlcov_suspect_lineup"],
            "description": "Running complete test suite with coverage"
        }
    ]
    
    results = {}
    
    for test_group in test_commands:
        print(f"\n{'-'*60}")
        print(f"Running: {test_group['name']}")
        print(f"Description: {test_group['description']}")
        print(f"Command: {' '.join(test_group['cmd'])}")
        print(f"{'-'*60}")
        
        try:
            result = subprocess.run(
                test_group['cmd'],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            results[test_group['name']] = {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            # Print output
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
                
            if result.returncode == 0:
                print(f"✅ {test_group['name']} PASSED")
            else:
                print(f"❌ {test_group['name']} FAILED (exit code: {result.returncode})")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_group['name']} TIMED OUT")
            results[test_group['name']] = {'returncode': -1, 'error': 'timeout'}
        except Exception as e:
            print(f"💥 {test_group['name']} ERROR: {str(e)}")
            results[test_group['name']] = {'returncode': -1, 'error': str(e)}
    
    # Summary
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    passed = 0
    failed = 0
    
    for name, result in results.items():
        status = "✅ PASSED" if result['returncode'] == 0 else "❌ FAILED"
        print(f"{name:40} {status}")
        
        if result['returncode'] == 0:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed + failed} | Passed: {passed} | Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All suspect lineup tests PASSED!")
        return 0
    else:
        print(f"\n⚠️  {failed} test group(s) FAILED")
        return 1

def run_specific_test_pattern(pattern):
    """Run tests matching a specific pattern."""
    cmd = [
        "python", "-m", "pytest", 
        f"tests/test_suspect_lineup*.py",
        "-v", "--tb=short",
        "-k", pattern
    ]
    
    print(f"Running tests matching pattern: {pattern}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    return result.returncode

def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Run specific test pattern
        pattern = sys.argv[1]
        return run_specific_test_pattern(pattern)
    else:
        # Run full test suite
        return run_tests()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 