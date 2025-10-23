from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
from flask_cors import CORS
import json
import os
import random
import sys
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw
import base64
import io

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production
CORS(app)

class AnnotationSystem:
    def __init__(self, user_folder_path: str = None):
        # Save files in outputs directory
        self.outputs_dir = os.path.join(os.getcwd(), "outputs")
        self.users_file = os.path.join(self.outputs_dir, "users.json")
        self.annotations_file = os.path.join(self.outputs_dir, "annotations.json")
        self.images_dir = None  # Will be set when user folder is loaded
        self.sample_images = []
        self.users = {}
        self.annotations = {}
        self.user_folder_path = user_folder_path  # Path to user folder (e.g., sft_splits/user1)
        self.user_metadata = {}  # Metadata for current user's images
        
        # Flag definitions with explanations
        self.flags = {
            "Shadows": "Check: Do all shadows point roughly the same way, show a dark contact at touchpoints, and get softer as they stretch away? PASS: All outdoor shadows lean the same way; tight dark line under shoes; edges soften outward. FAIL: One shadow points the opposite way; feet \"float\" (no dark contact); shadow edge equally sharp everywhere.",
            "Lighting Match": "Check: Are objects lit from the same side with similar brightness/contrast for the scene? PASS: Faces, walls, and props all brighter on the left side. FAIL: One object bright from the left while nearby objects are bright from the right.",
            "Color Cast Consistency": "Check: Do whites/neutrals share the same warm/cool tint? PASS: All whites slightly warm indoors. FAIL: A shirt is cold blue while everything else looks yellowish.",
            "Relative Size / Scale": "Check: Do familiar things get smaller with distance and look believably sized? PASS: Far people/cars noticeably smaller than near ones. FAIL: A far person is as big as someone close.",
            "Perspective / Straight-Line Convergence": "Check: Do parallel lines (road edges, table sides) aim toward the same point(s)? PASS: Building edges converge consistently. FAIL: An added sign's edges converge to a different point.",
            "Front–Back Overlap": "Check: Does the front object cleanly cover the back object without halos or see-through? PASS: Vase cleanly blocks the book behind it. FAIL: Background \"leaks\" through hair/edges; color fringe halo.",
            "Contact & Support": "Check: Do objects look supported and press into soft things when they should? PASS: Cushion dips under a person; grass bends under a foot. FAIL: Heavy box sits on a blanket with zero dent.",
            "Focus / Blur with Depth": "Check: Does blur increase smoothly with distance from the focus plane? PASS: Foreground soft → subject sharp → background gently soft. FAIL: A pasted object stays razor-sharp while neighbors at the same depth are blurry.",
            "Reflections & Transparency": "Check: Do reflections show the right thing at the right angle, and do transparent surfaces show both some reflection (at angles) and some see-through? PASS: Person appears (darker) in a shop window at the expected position; window shows faint outdoor reflection plus the room behind. FAIL: No reflection where expected, wrong pose/angle in mirror, or window is either a perfect mirror (no see-through) or perfectly clear (no reflection) at a glancing angle.",
            "Material Shine": "Check: Do shiny things have crisp highlights and matte things have soft, broad highlights? PASS: Metal = bright, sharp highlight; fabric = soft highlight. FAIL: Everything has the same plasticky shine.",
            "Texture / Pattern Follow-Through": "Check: Do stripes, weaves, wood grain bend and scale with folds and curves? PASS: Shirt stripes curve smoothly around the sleeve. FAIL: Pattern stays flat/smeared across bends.",
            "Text & Small Details": "Check: Are letters/numbers legible and small parts coherent? PASS: Street sign letters readable, normal spacing. FAIL: Gibberish letters, melted numbers, warped watch face markings.",
            "Object Completeness & Counts": "Check: Are obvious parts present and counted right (fingers, spokes, chair legs, petals, etc.)? PASS: Hand shows five distinct fingers; bicycle has two wheels and believable spokes. FAIL: Person has 3 fingers → FAIL here. Other fails: missing chair leg, duplicated petal, merged limbs.",
            "Edges & Boundaries (cut-out / halo check)": "Check: Do object edges look like they naturally belong in the scene? Edges should be as sharp/soft as nearby stuff at the same distance, with no glowing outline, no weird color rim, and no \"scissor-cut\" look. PASS: Jacket sleeve edge is slightly soft just like nearby edges; wispy hair/fur looks irregular (not clumped); no bright outline; edge grain looks like the background's grain. FAIL: A bright/dark \"halo\" around the object; a clean, too-straight cut line through fine detail; color fringe that doesn't match the rest of the photo; jaggy \"stair-steps\" only around the object; edge is razor-sharp while nearby, same-depth edges are blurry; hair melts into the background with a smeared border.",
            "Other": "Use this flag for any other inconsistencies or issues not covered by the specific flags above. Describe what you observe that seems unnatural, inconsistent, or problematic in the image."
        }
        
        # Create outputs directory if it doesn't exist
        if not os.path.exists(self.outputs_dir):
            os.makedirs(self.outputs_dir)
        
        self.load_data()
        
        # If user folder path is provided, load it automatically
        if self.user_folder_path:
            success, message = self.set_user_folder(self.user_folder_path)
            if not success:
                print(f"Error: Failed to load user folder '{self.user_folder_path}': {message}")
                print("Please provide a valid user folder path.")
                sys.exit(1)
    
    def set_user_folder(self, user_folder_path: str) -> Tuple[bool, str]:
        """Set the user folder path and load images from it"""
        if not os.path.exists(user_folder_path):
            return False, f"User folder does not exist: {user_folder_path}"
        
        # Check if it's a valid user folder structure
        images_dir = os.path.join(user_folder_path, "images")
        metadata_file = os.path.join(user_folder_path, "metadata.json")
        
        if not os.path.exists(images_dir):
            return False, f"Images directory not found in: {user_folder_path}"
        
        if not os.path.exists(metadata_file):
            return False, f"Metadata file not found in: {user_folder_path}"
        
        # Load metadata
        try:
            with open(metadata_file, 'r') as f:
                self.user_metadata = json.load(f)
        except Exception as e:
            return False, f"Error loading metadata: {str(e)}"
        
        # Set the user folder path and images directory
        self.user_folder_path = user_folder_path
        self.images_dir = images_dir
        
        # Load images from the user folder
        self.sample_images = []
        files = os.listdir(images_dir)
        
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                full_path = os.path.join(images_dir, file)
                self.sample_images.append(full_path)
        
        print(f"Loaded {len(self.sample_images)} images from user folder: {user_folder_path}")
        return True, f"Successfully loaded {len(self.sample_images)} images from user folder"
    
    def get_image_metadata(self, filename: str) -> Dict:
        """Get metadata for a specific image"""
        for metadata in self.user_metadata:
            if metadata.get('filename') == filename:
                return metadata
        return {}
    
    def get_image_display_name(self, filename: str) -> str:
        """Get display name for image showing type (edited/AI-generated)"""
        metadata = self.get_image_metadata(filename)
        image_type = metadata.get('type', 'unknown')
        
        if image_type == 'edited':
            return f"{filename} (EDITED)"
        elif image_type == 'fake':
            return f"{filename} (AI-GENERATED)"
        else:
            return filename
    
    def load_data(self):
        """Load existing users and annotations"""
        if os.path.exists(self.users_file):
            with open(self.users_file, 'r') as f:
                self.users = json.load(f)
        
        if os.path.exists(self.annotations_file):
            with open(self.annotations_file, 'r') as f:
                self.annotations = json.load(f)
    
    def save_data(self):
        """Save users and annotations to files"""
        print(f"Saving data to {self.users_file} and {self.annotations_file}")
        print(f"Annotations to save: {self.annotations}")
        
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2)
        
        with open(self.annotations_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        
        print("Data saved successfully to files")
    
    def register_user(self, name: str, email: str) -> Tuple[bool, str]:
        """Register a new user"""
        if not name or not email:
            return False, "Name and email are required!"
        
        if email in self.users:
            return False, f"User with email {email} already exists!"
        
        # In the new system, users don't get assigned specific images
        # Images are loaded from the user folder
        self.users[email] = {
            "name": name,
            "email": email,
            "registration_date": datetime.now().isoformat(),
            "last_annotated_image": None,
            "last_selected_flag": None
        }
        
        # Initialize empty annotations for this user
        self.annotations[email] = {}
        
        self.save_data()
        return True, f"User {name} registered successfully!"
    
    def login_user(self, email: str) -> Tuple[bool, str, Dict]:
        """Login user and return success status, message, and user data"""
        if email not in self.users:
            return False, "User not found. Please register first.", {}
        
        user_data = self.users[email]
        return True, f"Welcome back, {user_data['name']}!", user_data
    
    def get_image_annotations(self, email: str, img_name: str) -> Dict:
        """Get annotations for a specific image"""
        if email in self.annotations and img_name in self.annotations[email]:
            return self.annotations[email][img_name]
        return {"flags": {}, "last_updated": ""}
    
    def save_annotation(self, email: str, img_name: str, flag_name: str, bbox) -> Tuple[bool, str]:
        """Save annotation for a specific flag and image"""
        print(f"Saving annotation for {email}, {img_name}, {flag_name}, {bbox}")
        
        # Create user annotation entry if it doesn't exist
        if email not in self.annotations:
            self.annotations[email] = {}
            print(f"Created new user entry for {email}")
        
        if img_name not in self.annotations[email]:
            self.annotations[email][img_name] = {"flags": {}, "last_updated": ""}
            print(f"Created new image entry for {img_name}")
        
        # Initialize flag if it doesn't exist
        if flag_name not in self.annotations[email][img_name]["flags"]:
            self.annotations[email][img_name]["flags"][flag_name] = {
                "bboxes": [],
                "timestamp": datetime.now().isoformat()
            }
            print(f"Created new flag entry for {flag_name}")
        
        # Check if this is an update to an existing bbox (has referring expression)
        if isinstance(bbox, dict) and 'referringExpression' in bbox:
            print(f"Processing referring expression update: {bbox}")
            # This is an update to an existing bbox with referring expression
            bbox_index = bbox.get('bboxIndex', -1)
            print(f"Bbox index: {bbox_index}")
            if bbox_index >= 0 and bbox_index < len(self.annotations[email][img_name]["flags"][flag_name]["bboxes"]):
                # Update existing bbox with referring expression
                existing_bbox = self.annotations[email][img_name]["flags"][flag_name]["bboxes"][bbox_index]
                print(f"Existing bbox: {existing_bbox}")
                if isinstance(existing_bbox, list):
                    # Convert simple list to dict with referring expression
                    self.annotations[email][img_name]["flags"][flag_name]["bboxes"][bbox_index] = {
                        "coordinates": existing_bbox,
                        "ref_exp": bbox["referringExpression"]  # Use ref_exp as requested
                    }
                    print(f"Converted list to dict with ref_exp: {self.annotations[email][img_name]['flags'][flag_name]['bboxes'][bbox_index]}")
                else:
                    # Update existing dict
                    existing_bbox["ref_exp"] = bbox["referringExpression"]  # Use ref_exp as requested
                    print(f"Updated existing dict with ref_exp: {existing_bbox}")
                print(f"Updated bbox {bbox_index} with referring expression: {bbox['referringExpression']}")
            else:
                print(f"Warning: Invalid bbox index {bbox_index} for update")
                return False, f"Invalid bounding box index for update"
        else:
            # This is a new bbox - check if it's already in the new format
            if isinstance(bbox, dict) and 'coordinates' in bbox and 'ref_exp' in bbox:
                # Already in new format, just add it
                self.annotations[email][img_name]["flags"][flag_name]["bboxes"].append(bbox)
                print(f"Added new bbox in new format to {flag_name}, total bboxes: {len(self.annotations[email][img_name]['flags'][flag_name]['bboxes'])}")
            else:
                # Convert old format to new format with empty ref_exp
                new_bbox = {
                    "coordinates": bbox,
                    "ref_exp": ""
                }
                self.annotations[email][img_name]["flags"][flag_name]["bboxes"].append(new_bbox)
                print(f"Converted and added new bbox to {flag_name} with empty ref_exp, total bboxes: {len(self.annotations[email][img_name]['flags'][flag_name]['bboxes'])}")
        
        self.annotations[email][img_name]["flags"][flag_name]["timestamp"] = datetime.now().isoformat()
        self.annotations[email][img_name]["last_updated"] = datetime.now().isoformat()
        
        # Update last annotated image and last selected flag for the user
        if email in self.users:
            self.users[email]["last_annotated_image"] = img_name
            self.users[email]["last_selected_flag"] = flag_name
        
        # Save to file immediately
        print("Saving data to file...")
        print(f"Final annotations structure for {email}, {img_name}: {self.annotations[email][img_name]}")
        self.save_data()
        print("Data saved successfully")
        
        return True, f"Bounding box {'updated' if isinstance(bbox, dict) else 'added'} for {flag_name}!"
    
    def update_referring_expression(self, email: str, img_name: str, flag_name: str, bbox_index: int, referring_expression: str) -> Tuple[bool, str]:
        """Update referring expression for an existing bounding box"""
        print(f"Updating referring expression for {email}, {img_name}, {flag_name}, bbox {bbox_index}: {referring_expression}")
        
        # Create user annotation entry if it doesn't exist
        if email not in self.annotations:
            self.annotations[email] = {}
            print(f"Created new user entry for {email}")
        
        if img_name not in self.annotations[email]:
            self.annotations[email][img_name] = {"flags": {}, "last_updated": ""}
            print(f"Created new image entry for {img_name}")
        
        if flag_name not in self.annotations[email][img_name]["flags"]:
            print(f"Warning: Flag {flag_name} not found for {img_name}")
            return False, f"Flag {flag_name} not found"
        
        bboxes = self.annotations[email][img_name]["flags"][flag_name]["bboxes"]
        if bbox_index < 0 or bbox_index >= len(bboxes):
            print(f"Warning: Invalid bbox index {bbox_index}")
            return False, f"Invalid bounding box index {bbox_index}"
        
        # Update the referring expression
        existing_bbox = bboxes[bbox_index]
        if isinstance(existing_bbox, list):
            # Convert old format to new format
            bboxes[bbox_index] = {
                "coordinates": existing_bbox,
                "ref_exp": referring_expression
            }
            print(f"Converted bbox {bbox_index} from list to dict with ref_exp: {referring_expression}")
        else:
            # Update existing format
            existing_bbox["ref_exp"] = referring_expression
            print(f"Updated bbox {bbox_index} with ref_exp: {referring_expression}")
        
        # Update timestamps
        self.annotations[email][img_name]["flags"][flag_name]["timestamp"] = datetime.now().isoformat()
        self.annotations[email][img_name]["last_updated"] = datetime.now().isoformat()
        
        # Save to file immediately
        print("Saving referring expression update to file...")
        self.save_data()
        print("Referring expression update saved successfully")
        
        return True, f"Referring expression updated for {flag_name}!"
    
    def remove_annotation(self, email: str, img_name: str, flag_name: str, bbox_index: int = None) -> Tuple[bool, str]:
        """Remove annotation for a specific flag and image"""
        # Create user annotation entry if it doesn't exist
        if email not in self.annotations:
            self.annotations[email] = {}
        
        if img_name not in self.annotations[email]:
            self.annotations[email][img_name] = {"flags": {}, "last_updated": ""}
        
        if (img_name in self.annotations[email] and 
            flag_name in self.annotations[email][img_name]["flags"]):
            
            if bbox_index is not None:
                # Remove specific bounding box
                bboxes = self.annotations[email][img_name]["flags"][flag_name]["bboxes"]
                if 0 <= bbox_index < len(bboxes):
                    bboxes.pop(bbox_index)
                    if not bboxes:  # If no more bboxes, remove the flag entirely
                        del self.annotations[email][img_name]["flags"][flag_name]
                else:
                    return False, f"Invalid bounding box index for {flag_name}"
            else:
                # Remove entire flag
                del self.annotations[email][img_name]["flags"][flag_name]
            
            self.annotations[email][img_name]["last_updated"] = datetime.now().isoformat()
            self.save_data()
            return True, f"Annotation removed for {flag_name}!"
        
        return False, f"No annotation found for {flag_name}!"
    
    def update_last_selected_flag(self, email: str, flag_name: str) -> bool:
        """Update the last selected flag for a user"""
        if email in self.users:
            self.users[email]["last_selected_flag"] = flag_name
            self.save_data()
            return True
        return False

# Initialize the annotation system (will be reinitialized with user folder if provided)
annotation_system = AnnotationSystem()

@app.route('/')
def index():
    if 'user_email' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        success, message, user_data = annotation_system.login_user(email)
        
        if success:
            session['user_email'] = email
            session['user_name'] = user_data['name']
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error=message)
    
    return render_template('login.html')

@app.route('/select_user_folder', methods=['GET', 'POST'])
def select_user_folder():
    """Route to select user folder for annotation"""
    if request.method == 'POST':
        user_folder_path = request.form.get('user_folder_path')
        if user_folder_path:
            success, message = annotation_system.set_user_folder(user_folder_path)
            if success:
                session['user_folder_path'] = user_folder_path
                return redirect(url_for('dashboard'))
            else:
                return render_template('select_user_folder.html', error=message)
        else:
            return render_template('select_user_folder.html', error="Please provide a user folder path")
    
    return render_template('select_user_folder.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        success, message = annotation_system.register_user(name, email)
        
        if success:
            return render_template('login.html', success=message)
        else:
            return render_template('register.html', error=message)
    
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'user_email' not in session:
        return redirect(url_for('login'))
    
    user_email = session['user_email']
    user_data = annotation_system.users.get(user_email, {})
    
    # Check if we have a user folder selected (either from session or from command line)
    if 'user_folder_path' in session:
        user_images = annotation_system.sample_images
        user_folder_path = session['user_folder_path']
    elif annotation_system.user_folder_path:
        user_images = annotation_system.sample_images
        user_folder_path = annotation_system.user_folder_path
    else:
        return redirect(url_for('select_user_folder'))
    
    # Get annotation data for all user images
    user_annotations = {}
    image_metadata = {}
    for img_path in user_images:
        img_name = os.path.basename(img_path)
        user_annotations[img_name] = annotation_system.get_image_annotations(user_email, img_name)
        image_metadata[img_name] = annotation_system.get_image_metadata(img_name)
    
    # Store annotations in session for template access
    session['annotations'] = user_annotations
    
    # Check if user explicitly wants to see dashboard (not auto-redirect)
    show_dashboard = request.args.get('show', 'false').lower() == 'true'
    
    # Only auto-redirect if user hasn't explicitly requested dashboard
    if not show_dashboard and user_data.get('last_annotated_image'):
        return redirect(url_for('annotate', image_name=user_data['last_annotated_image']))
    
    return render_template('dashboard.html', 
                         user_name=session['user_name'],
                         user_images=user_images,
                         annotations=user_annotations,
                         image_metadata=image_metadata,
                         user_folder_path=user_folder_path)

@app.route('/annotate/<image_name>')
def annotate(image_name):
    if 'user_email' not in session:
        return redirect(url_for('login'))
    
    user_email = session['user_email']
    
    # Check if we have a user folder selected (either from session or from command line)
    if 'user_folder_path' in session:
        user_images = annotation_system.sample_images
    elif annotation_system.user_folder_path:
        user_images = annotation_system.sample_images
    else:
        return redirect(url_for('select_user_folder'))
    
    # Check if user has access to this image
    if not any(image_name in img for img in user_images):
        return redirect(url_for('dashboard'))
    
    # Get image path
    img_path = next(img for img in user_images if image_name in img)
    annotations = annotation_system.get_image_annotations(user_email, image_name)
    
    # Get image metadata
    image_metadata = annotation_system.get_image_metadata(image_name)
    
    # Get display name for the image
    display_name = annotation_system.get_image_display_name(image_name)
    
    # Get last selected flag for the user
    last_selected_flag = annotation_system.users.get(user_email, {}).get('last_selected_flag')
    
    return render_template('annotate.html',
                         user_name=session['user_name'],
                         image_name=image_name,
                         display_name=display_name,
                         image_path=img_path,
                         annotations=annotations,
                         image_metadata=image_metadata,
                         flags=annotation_system.flags,
                         last_selected_flag=last_selected_flag)

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve images from the configured images directory"""
    return send_from_directory(annotation_system.images_dir, filename)

@app.route('/api/save_annotation', methods=['POST'])
def api_save_annotation():
    if 'user_email' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    data = request.get_json()
    user_email = session['user_email']
    img_name = data.get('image_name')
    flag_name = data.get('flag_name')
    bbox = data.get('bbox')
    
    if not all([img_name, flag_name, bbox]):
        return jsonify({'success': False, 'message': 'Missing required data'})
    
    success, message = annotation_system.save_annotation(user_email, img_name, flag_name, bbox)
    
    # Update session annotations after saving
    if success:
        user_annotations = session.get('annotations', {})
        if img_name not in user_annotations:
            user_annotations[img_name] = {"flags": {}, "last_updated": ""}
        
        if flag_name not in user_annotations[img_name]["flags"]:
            user_annotations[img_name]["flags"][flag_name] = {"bboxes": [], "timestamp": ""}
        
        user_annotations[img_name]["flags"][flag_name]["bboxes"].append(bbox)
        user_annotations[img_name]["flags"][flag_name]["timestamp"] = datetime.now().isoformat()
        user_annotations[img_name]["last_updated"] = datetime.now().isoformat()
        session['annotations'] = user_annotations
    
    return jsonify({'success': success, 'message': message})

@app.route('/api/update_referring_expression', methods=['POST'])
def api_update_referring_expression():
    """Update referring expression for an existing bounding box"""
    if 'user_email' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    data = request.get_json()
    user_email = session['user_email']
    img_name = data.get('image_name')
    flag_name = data.get('flag_name')
    bbox_index = data.get('bbox_index')
    referring_expression = data.get('referring_expression')
    
    if not all([img_name, flag_name, bbox_index is not None, referring_expression]):
        return jsonify({'success': False, 'message': 'Missing required data'})
    
    # Update the referring expression in the backend
    success, message = annotation_system.update_referring_expression(
        user_email, img_name, flag_name, bbox_index, referring_expression
    )
    
    # Update session annotations after saving
    if success:
        user_annotations = session.get('annotations', {})
        if (img_name in user_annotations and 
            flag_name in user_annotations[img_name]["flags"] and
            bbox_index < len(user_annotations[img_name]["flags"][flag_name]["bboxes"])):
            
            bbox = user_annotations[img_name]["flags"][flag_name]["bboxes"][bbox_index]
            if isinstance(bbox, list):
                # Convert old format to new format
                user_annotations[img_name]["flags"][flag_name]["bboxes"][bbox_index] = {
                    "coordinates": bbox,
                    "ref_exp": referring_expression
                }
            else:
                # Update existing format
                bbox["ref_exp"] = referring_expression
            
            user_annotations[img_name]["flags"][flag_name]["timestamp"] = datetime.now().isoformat()
            user_annotations[img_name]["last_updated"] = datetime.now().isoformat()
            session['annotations'] = user_annotations
    
    return jsonify({'success': success, 'message': message})

@app.route('/api/remove_annotation', methods=['POST'])
def api_remove_annotation():
    if 'user_email' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    data = request.get_json()
    user_email = session['user_email']
    img_name = data.get('image_name')
    flag_name = data.get('flag_name')
    bbox_index = data.get('bbox_index')
    
    if not all([img_name, flag_name]):
        return jsonify({'success': False, 'message': 'Missing required data'})
    
    success, message = annotation_system.remove_annotation(user_email, img_name, flag_name, bbox_index)
    
    # Update session annotations after removing
    if success:
        user_annotations = session.get('annotations', {})
        if img_name in user_annotations and flag_name in user_annotations[img_name]["flags"]:
            if bbox_index is not None:
                # Remove specific bounding box
                bboxes = user_annotations[img_name]["flags"][flag_name]["bboxes"]
                if 0 <= bbox_index < len(bboxes):
                    bboxes.pop(bbox_index)
                    if not bboxes:  # If no more bboxes, remove the flag entirely
                        del user_annotations[img_name]["flags"][flag_name]
            else:
                # Remove entire flag
                del user_annotations[img_name]["flags"][flag_name]
            
            user_annotations[img_name]["last_updated"] = datetime.now().isoformat()
            session['annotations'] = user_annotations
    
    return jsonify({'success': True, 'message': message})

@app.route('/api/get_annotations/<image_name>')
def api_get_annotations(image_name):
    if 'user_email' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    user_email = session['user_email']
    annotations = annotation_system.get_image_annotations(user_email, image_name)
    return jsonify({'success': True, 'annotations': annotations})

@app.route('/api/navigate/<direction>/<image_name>')
def navigate_image(direction, image_name):
    """Navigate to next or previous image"""
    if 'user_email' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    user_email = session['user_email']
    user_images = annotation_system.get_user_images(user_email)
    
    # Find current image index
    current_index = -1
    for i, img_path in enumerate(user_images):
        if image_name in img_path:
            current_index = i
            break
    
    if current_index == -1:
        return jsonify({'success': False, 'message': 'Image not found'})
    
    # Calculate next/previous index
    if direction == 'next':
        next_index = (current_index + 1) % len(user_images)
    elif direction == 'previous':
        next_index = (current_index - 1) % len(user_images)
    else:
        return jsonify({'success': False, 'message': 'Invalid direction'})
    
    next_image_name = os.path.basename(user_images[next_index])
    return jsonify({'success': True, 'next_image': next_image_name})

@app.route('/api/update_last_flag', methods=['POST'])
def api_update_last_flag():
    if 'user_email' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    data = request.get_json()
    user_email = session['user_email']
    flag_name = data.get('flag_name')
    
    if not flag_name:
        return jsonify({'success': False, 'message': 'Missing flag name'})
    
    success = annotation_system.update_last_selected_flag(user_email, flag_name)
    return jsonify({'success': success, 'message': 'Flag updated' if success else 'Failed to update flag'})

@app.route('/api/refresh_annotations')
def api_refresh_annotations():
    if 'user_email' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    user_email = session['user_email']
    
    # Get user images from the loaded user folder
    if annotation_system.user_folder_path:
        user_images = annotation_system.sample_images
    else:
        return jsonify({'success': False, 'message': 'No user folder loaded'})
    
    # Get updated annotation data for all user images
    user_annotations = {}
    for img_path in user_images:
        img_name = os.path.basename(img_path)
        user_annotations[img_name] = annotation_system.get_image_annotations(user_email, img_name)
    
    # Update session annotations
    session['annotations'] = user_annotations
    
    return jsonify({'success': True, 'annotations': user_annotations})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Annotation System')
    parser.add_argument('--user-folder', '-u', type=str, help='Path to user folder (e.g., sft_splits/user1)')
    parser.add_argument('--host', default='0.0.0.0', help='Host to run the app on')
    parser.add_argument('--port', type=int, default=7865, help='Port to run the app on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Reinitialize annotation system with user folder (required)
    if args.user_folder:
        print(f"Loading user folder: {args.user_folder}")
        annotation_system = AnnotationSystem(user_folder_path=args.user_folder)
        print(f"Successfully loaded {len(annotation_system.sample_images)} images from user folder")
    else:
        print("Error: User folder is required!")
        print("Usage: python app.py --user-folder sft_splits/user1")
        sys.exit(1)
    
    app.run(host=args.host, port=args.port, debug=args.debug)
