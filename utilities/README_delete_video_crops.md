# Video Crop Deletion Utility

A comprehensive utility to safely delete all crops from a specific video while preserving labeled images. This tool provides multiple safety checks, detailed logging, and backup capabilities.

## Features

- ✅ **Safe by default**: Runs in dry-run mode unless explicitly told to execute
- 🔍 **Comprehensive scanning**: Finds crops in all standard directories (`crow_crops`, `videos`, `processing`, etc.)  
- 🏷️ **Label preservation**: Automatically preserves crops that have been manually labeled
- 📊 **Detailed analysis**: Shows exactly what will be deleted before execution
- 💾 **Database cleanup**: Removes corresponding embeddings from the database
- 📄 **Backup lists**: Creates JSON backup files for recovery purposes
- 🧹 **Directory cleanup**: Removes empty directories after deletion
- ⚡ **Flexible matching**: Matches crops by video name, path, or filename patterns

## Safety Features

1. **Dry Run Default**: All operations are simulated by default
2. **User Confirmation**: Asks for explicit confirmation before deleting files
3. **Label Protection**: Preserves images with labels unless explicitly overridden
4. **Backup Creation**: Creates detailed backup lists of deleted files
5. **Error Handling**: Continues operation even if individual files fail
6. **Detailed Logging**: Comprehensive logging of all operations

## Usage

### Basic Usage (Safe Preview)
```bash
# Preview what would be deleted (safe)
python utilities/delete_video_crops.py video123.mp4
```

### Delete Unlabeled Crops Only
```bash
# Delete unlabeled crops, preserve labeled ones
python utilities/delete_video_crops.py video123.mp4 --execute
```

### Delete All Crops (Including Labeled)
```bash
# Delete ALL crops from video (dangerous!)
python utilities/delete_video_crops.py video123.mp4 --execute --no-preserve-labeled
```

### Command Line Options

- `--execute`: Actually perform the deletion (default is dry-run)
- `--no-preserve-labeled`: Delete labeled images too (dangerous)
- `--dry-run`: Force dry-run mode (explicit safety)

## Example Output

```
🔍 Analyzing crops for video: my_video
Scanning directory: crow_crops
Scanning directory: videos
Found 245 potential video-related crops

📊 ANALYSIS SUMMARY
==================================================
Video: my_video
Total crops found: 245
Labeled crops: 23
Unlabeled crops: 222
Crops to delete: 222
Crops to preserve: 23

📋 PRESERVED CROPS (Labeled):
  crow: 18 images
  not_a_crow: 3 images
  bad_crow: 2 images

🗑️ CROPS TO DELETE:
  crow_crops: 180 images
  videos: 42 images

⚠️  This will delete 222 crop files and their database entries. Continue? (y/N):
```

## What Gets Deleted

The utility will find and delete:

### File Locations
- `crow_crops/` - Main crop directory
- `crow_crops2/` - Secondary crop directory  
- `videos/` - Video-organized crops
- `processing/` - Processing directories
- `non_crow_crops/` - Non-crow bird crops
- `potential_not_crow_crops/` - Low confidence crops
- `hard_negatives/` - Hard negative examples
- `false_positive_crops/` - False positive crops

### Database Entries
- Crow embeddings associated with the deleted crop files
- Updates crow statistics and sighting counts
- Maintains database integrity

### Filename Patterns
The utility matches crops using flexible patterns:
- Direct video name matches
- Video stem (filename without extension)
- Cleaned names (spaces/dots converted to underscores)

## What Gets Preserved

By default, the utility preserves:

- **Labeled Images**: Any crop with a manual label (`crow`, `not_a_crow`, `bad_crow`, `not_sure`, `multi_crow`)
- **Training Data**: Images marked as training data in the database
- **Directory Structure**: Parent directories that still contain files

## Backup and Recovery

The utility automatically creates backup files:

```json
{
  "video_name": "my_video",
  "video_path": "my_video.mp4", 
  "deletion_timestamp": "20241227_143022",
  "dry_run": false,
  "preserve_labeled": true,
  "stats": {
    "files_deleted": 222,
    "embeddings_deleted": 189,
    "crops_preserved": 23
  },
  "deleted_crops": [...],
  "preserved_crops": [...]
}
```

## Examples

### Interactive Example
```bash
python utilities/example_delete_video_crops.py interactive
```

### Batch Processing Multiple Videos
```python
from utilities.delete_video_crops import VideoCropDeletor

videos_to_clean = ["video1.mp4", "video2.mp4", "video3.mp4"]

for video in videos_to_clean:
    deletor = VideoCropDeletor(
        video_path=video,
        dry_run=False,
        preserve_labeled=True
    )
    deletor.execute_deletion()
```

### Custom Analysis Only
```python
from utilities.delete_video_crops import VideoCropDeletor

deletor = VideoCropDeletor("problem_video.mp4", dry_run=True)
deletor.analyze_video_crops()  # Just analyze, don't delete

print(f"Found {deletor.stats['total_crops_found']} crops")
print(f"Would delete {deletor.stats['crops_to_delete']} crops")
```

## Error Handling

The utility is designed to be robust:

- **File Not Found**: Continues processing other files
- **Permission Errors**: Logs error but continues
- **Database Errors**: Attempts to continue with file deletion
- **Metadata Corruption**: Skips corrupted metadata files
- **Keyboard Interrupt**: Graceful shutdown with status

## Best Practices

1. **Always dry-run first**: Review what will be deleted
2. **Check labels**: Ensure important crops are labeled before deletion
3. **Backup your database**: Create database backups before large deletions
4. **Monitor logs**: Check for errors during execution
5. **Verify results**: Confirm expected crops were removed

## Integration with Labeling Workflow

This utility integrates with the existing labeling system:

1. Label important crops using the image reviewer
2. Run dry-run to preview deletion
3. Execute deletion to clean up unlabeled crops
4. Continue with clean dataset for training

## Troubleshooting

### No Crops Found
- Check video name spelling
- Verify crop directories exist
- Look for filename pattern mismatches

### Permission Errors
- Run with appropriate file permissions
- Check if files are open in other applications
- Verify disk space and access rights

### Database Connection Issues
- Ensure database file exists and is accessible
- Check database permissions
- Verify database isn't corrupted

## Technical Details

### Database Operations
- Uses `get_embedding_ids_by_image_paths()` to find related embeddings
- Calls `delete_crow_embeddings()` to remove database entries
- Updates crow statistics and sighting counts automatically

### File Matching Algorithm
1. Extract video stem from input path
2. Generate multiple search patterns
3. Search recursively in all crop directories
4. Match against filename patterns
5. Cross-reference with metadata files

### Safety Mechanisms
- Multiple confirmation prompts
- Extensive logging and error handling
- Backup file creation before deletion
- Database transaction safety
- Graceful error recovery 