# Virtual Try-On Demo

A web-based virtual try-on application that allows users to upload person and clothing images to see realistic virtual try-on results.

## Requirements
- Python 3.7+
- PyTorch
- Flask
- Other dependencies listed in `requirements.txt`

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd virtual-tryon-demo
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Download pre-trained models:
   - Place the model checkpoints in the `checkpoints/` directory:
     - `seg_final.pth`
     - `gmm_final.pth`
     - `alias_final.pth`
6. Prepare dataset:
   - Ensure the test dataset is in the `datasets/test/` directory with the following structure:
     ```
     datasets/
     └── test/
         ├── image/          # Person images
         ├── cloth/          # Clothing images
         ├── cloth-mask/     # Clothing masks
         ├── image-parse/    # Person parsing
         └── pose/           # Pose data
     ```

## Usage
1. Start the application:
   ```bash
   python app.py
   ```
2. Open your browser and go to `http://localhost:5000`
3. Upload images:
   - Upload a person image from the test dataset (e.g., "08909_00.jpg")
   - Upload a clothing image from the test dataset (e.g., "01430_00.jpg")
4. Generate results:
   - Click "Generate Virtual Try-On" to process the images
   - Wait for the processing to complete
   - View the virtual try-on result

## Notes
- The application currently works with images from the test dataset only
- Upload images with the exact filenames from the test dataset
- The first run may take longer as models are loaded into memory
- Ensure sufficient RAM for model loading and inference
