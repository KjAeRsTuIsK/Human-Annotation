# Image Annotation System

A Flask-based web application for annotating images with bounding boxes and flagging inconsistencies.

## Features

- **User Folder Support**: Load images from specific user folders (e.g., sft_splits/user1)
- **Image Categorization**: Automatically categorizes images as "EDITED" or "AI-GENERATED"
- **Metadata Display**: Shows descriptions and edit instructions for edited images
- **Flagging System**: 14 different flags for detecting various inconsistencies
- **Bounding Box Annotation**: Draw bounding boxes and add referring expressions
- **Session Management**: User login/registration system

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python app.py --user-folder sft_splits/user1
```

### Advanced Usage
```bash
python app.py --user-folder sft_splits/user2 --port 8080 --debug
```

### Command Line Arguments
- `--user-folder` or `-u`: Path to user folder (required)
- `--host`: Host to run on (default: 0.0.0.0)
- `--port`: Port to run on (default: 7865)
- `--debug`: Run in debug mode

## User Folder Structure

The user folder should contain:
```
user_folder/
├── images/          # Image files (.jpg, .png, etc.)
└── metadata.json    # Metadata for each image
```

## Metadata Format

Each image in metadata.json should have:
```json
{
  "image_path": "/path/to/image.jpg",
  "type": "edited" or "fake",
  "source": "dataset_name",
  "filename": "image.jpg",
  "description": "Description for edited images",
  "instruction": {
    "Effect": "Effect description",
    "Change Target": "Target description",
    "Explanation": ["Explanation items"]
  }
}
```

## Image Types

- **edited**: Images that have been manually edited/tampered
- **fake**: AI-generated or synthetic images

## Flags

The system includes 14 flags for detecting inconsistencies:
1. Shadows
2. Lighting Match
3. Color Cast Consistency
4. Relative Size / Scale
5. Perspective / Straight-Line Convergence
6. Front–Back Overlap
7. Contact & Support
8. Focus / Blur with Depth
9. Reflections & Transparency
10. Material Shine
11. Texture / Pattern Follow-Through
12. Text & Small Details
13. Object Completeness & Counts
14. Edges & Boundaries (cut-out / halo check)
15. Other

## Web Interface

1. Open your browser and go to `http://localhost:7865`
2. Register a new user or login with existing credentials
3. Start annotating images by selecting flags and drawing bounding boxes
4. Add referring expressions to describe what each bounding box refers to

## Output

Annotations are saved to:
- `outputs/users.json`: User information
- `outputs/annotations.json`: All annotations with bounding boxes and referring expressions
