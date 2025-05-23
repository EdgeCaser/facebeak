# Suspect Lineup Feature - Test Coverage Documentation

## Overview

This document describes the comprehensive test coverage for the Suspect Lineup feature, which allows users to verify and correct crow ID assignments through a GUI interface.

## Test Structure

The test suite is organized into three main categories:

### 1. Database Function Tests (`test_suspect_lineup_db.py`)
**Coverage: Database operations and fallback logic**

#### Key Functions Tested:
- `get_crow_videos()` - Normal and fallback modes
- `get_first_crow_image()` - With and without embeddings
- `get_crow_images_from_video()` - Fallback when no embeddings exist
- `get_embedding_ids_by_image_paths()` - Image path to embedding ID mapping
- `delete_crow_embeddings()` - Embedding deletion
- `reassign_crow_embeddings()` - Moving embeddings between crows
- `create_new_crow_from_embeddings()` - Creating new crows from existing data

#### Test Scenarios:
- ✅ Normal operation with embeddings in database
- ✅ Fallback mode when no embeddings exist but crop images do
- ✅ Edge cases: nonexistent crows, empty lists, permission errors
- ✅ Database integrity after operations
- ✅ Filename parsing logic for image-to-embedding mapping

### 2. GUI Component Tests (`test_suspect_lineup_gui.py`)
**Coverage: User interface components and interactions**

#### Key Components Tested:
- `SuspectLineupApp` initialization
- Crow loading and dropdown population
- Image display and classification
- Video selection and filtering
- Save/discard operations
- Dialog interactions

#### Test Scenarios:
- ✅ GUI initialization and widget creation
- ✅ Loading crows into dropdown
- ✅ Displaying crow images and handling missing files
- ✅ User classification tracking
- ✅ Save operations (create new crow vs move to existing)
- ✅ Input validation and error handling
- ✅ Window closing with unsaved changes

### 3. Integration Tests (`test_suspect_lineup_integration.py`)
**Coverage: End-to-end workflows and component interactions**

#### Key Workflows Tested:
- Complete suspect lineup workflow from start to finish
- Database sync utility integration
- GUI launcher integration
- Error handling across components

#### Test Scenarios:
- ✅ Full workflow: load crows → select crow → classify images → save changes
- ✅ Database sync with real crop directory structure
- ✅ Fallback mode integration testing
- ✅ Cross-component error propagation
- ✅ Edge cases in integrated environment

## Test Coverage Metrics

### Target Coverage Goals:
- **Database Functions**: 95%+ line coverage
- **GUI Components**: 85%+ line coverage (UI testing limitations)
- **Integration Workflows**: 90%+ scenario coverage

### Key Areas Covered:
1. **Happy Path**: Normal user workflows
2. **Edge Cases**: Empty data, missing files, invalid inputs
3. **Error Handling**: Database errors, file system issues, GUI exceptions
4. **Fallback Logic**: Database/filesystem mismatches
5. **Data Integrity**: Ensuring operations don't corrupt data

## Running Tests

### Quick Start
```bash
# Run all suspect lineup tests
python run_suspect_lineup_tests.py

# Run specific test pattern
python run_suspect_lineup_tests.py "test_database_sync"

# Run individual test files
pytest tests/test_suspect_lineup_db.py -v
pytest tests/test_suspect_lineup_gui.py -v
pytest tests/test_suspect_lineup_integration.py -v
```

### Advanced Testing
```bash
# Run with coverage report
pytest tests/test_suspect_lineup*.py --cov=suspect_lineup --cov=sync_database --cov-report=html

# Run only database tests
pytest -m "suspect_lineup and database" -v

# Run only GUI tests
pytest -m "suspect_lineup and gui" -v

# Run only integration tests
pytest -m "integration" tests/test_suspect_lineup*.py -v
```

## Mock Usage and Test Isolation

### Database Mocking
- Each test uses a temporary SQLite database
- Database paths are patched to avoid interfering with real data
- Comprehensive cleanup ensures no test pollution

### GUI Mocking
- Tkinter widgets are fully mocked to enable headless testing
- PIL image operations are mocked to avoid file dependencies
- Database operations are mocked to test GUI logic in isolation

### File System Mocking
- Crop directory access is mocked for predictable testing
- Image file existence is controlled through mock patches
- Temporary directories used for integration tests

## Test Data Scenarios

### Database Test Data:
- Multiple crows with varying embedding counts
- Mixed video types (MP4, MOV, AVI)
- Different frame counts and detection numbers
- Edge cases: empty crows, single sightings

### GUI Test Data:
- Various classification scenarios
- Multiple image selections
- Different user interaction patterns
- Error conditions and recovery

### Integration Test Data:
- Comprehensive crop directory structure
- Real-world filename patterns
- Multiple crow scenarios with different sighting counts

## Continuous Integration Considerations

### Test Performance:
- Database tests: ~10-15 seconds
- GUI tests: ~20-30 seconds (widget mocking overhead)
- Integration tests: ~30-45 seconds
- **Total runtime**: ~60-90 seconds

### Dependencies:
- All tests run without requiring actual video files
- No external dependencies beyond standard test libraries
- Headless operation suitable for CI/CD

### Parallel Execution:
- Tests are designed for parallel execution
- Database isolation prevents conflicts
- Temporary directories ensure file system isolation

## Debugging Failed Tests

### Common Issues:
1. **Database Lock Errors**: Ensure proper connection cleanup
2. **Mock Setup Failures**: Check all required mocks are in place
3. **Temporary File Cleanup**: Windows file locking can cause issues

### Debug Commands:
```bash
# Run with maximum verbosity
pytest tests/test_suspect_lineup_db.py -vvv --tb=long

# Run single test method
pytest tests/test_suspect_lineup_db.py::TestSuspectLineupDatabase::test_get_crow_videos_fallback_mode -v

# Run with pdb on failure
pytest tests/test_suspect_lineup_gui.py --pdb
```

## Coverage Gaps and Future Improvements

### Current Limitations:
- PIL image manipulation testing is mocked (not testing actual image processing)
- Some edge cases in GUI event handling are difficult to test
- Performance testing under load not included

### Future Enhancements:
- Add property-based testing for database operations
- Include visual regression testing for GUI components
- Add performance benchmarks for large crop directories
- Test actual image processing pipeline

## Contributing to Tests

### Adding New Tests:
1. Follow existing naming conventions
2. Include comprehensive docstrings
3. Test both happy path and edge cases
4. Ensure proper cleanup in tearDown methods
5. Add appropriate pytest markers

### Test Quality Guidelines:
- Each test should test one specific behavior
- Tests should be independent and not rely on execution order
- Mock external dependencies appropriately
- Include meaningful assertions with descriptive messages
- Aim for clear, readable test code 