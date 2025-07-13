# Virtual Try-On Demo

A web-based virtual try-on application that allows users to upload person and clothing images to see realistic virtual try-on results.

## Features

- **Web Interface**: Modern, responsive UI for easy image upload
- **Real-time Processing**: Upload images and get results directly in the browser
- **High-Quality Results**: Uses pre-trained models for realistic virtual try-on
- **Drag & Drop**: Easy file upload with drag and drop support
- **Mobile Responsive**: Works on desktop and mobile devices

## How It Works

1. **Upload Images**: Upload a person image and a clothing item image
2. **Image Processing**: The system extracts image numbers from filenames
3. **Model Inference**: Uses pre-trained models to generate virtual try-on results
4. **Display Results**: Shows the final virtual try-on image in the browser

## Requirements

- Python 3.7+
- PyTorch
- Flask
- Other dependencies listed in `requirements.txt`

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd virtual-tryon-demo
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Download pre-trained models**:
   - Place the model checkpoints in the `checkpoints/` directory:
     - `seg_final.pth`
     - `gmm_final.pth`
     - `alias_final.pth`

6. **Prepare dataset**:
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

1. **Start the application**:
   ```bash
   python app.py
   ```

2. **Open your browser** and go to `http://localhost:5000`

3. **Upload images**:
   - Upload a person image from the test dataset (e.g., "08909_00.jpg")
   - Upload a clothing image from the test dataset (e.g., "01430_00.jpg")

4. **Generate results**:
   - Click "Generate Virtual Try-On" to process the images
   - Wait for the processing to complete
   - View the virtual try-on result

## Important Notes

- **Test Images Only**: The application currently works with images from the test dataset only
- **Filename Format**: Upload images with the exact filenames from the test dataset
- **Processing Time**: The first run may take longer as models are loaded into memory
- **Memory Requirements**: Ensure sufficient RAM for model loading and inference

## File Structure

```
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Web interface
├── datasets/
│   └── test/             # Test dataset
├── checkpoints/          # Pre-trained models
├── results/              # Generated results
└── uploads/              # Temporary upload storage
```

## API Endpoints

- `GET /`: Main web interface
- `POST /upload`: Upload and process images
- `GET /results/<filename>`: Serve result images

## Troubleshooting

### Common Issues

1. **Module not found errors**:
   - Ensure virtual environment is activated
   - Install all requirements: `pip install -r requirements.txt`

2. **Model loading errors**:
   - Check that model files exist in `checkpoints/` directory
   - Verify model file names match expected names

3. **Dataset errors**:
   - Ensure test dataset structure is correct
   - Check that uploaded images exist in the dataset

4. **Memory errors**:
   - Close other applications to free memory
   - Consider using a machine with more RAM

## Development

### Adding New Features

1. **Frontend**: Modify `templates/index.html` for UI changes
2. **Backend**: Edit `app.py` for server-side logic
3. **Models**: Update model loading in the `load_models()` function

### Testing

- Test with different image pairs from the test dataset
- Verify error handling with invalid uploads
- Check mobile responsiveness

## License

This project is for demonstration purposes. Please respect the original model licenses and dataset terms of use.

## Acknowledgments

- Based on VITON-HD research and implementation
- Uses pre-trained models for virtual try-on generation
- Web interface built with Flask and modern CSS
