import os
import random
from flask import Flask, send_from_directory, render_template, request
import torch
from datasets import VITONDataset
from networks import SegGenerator, GMM, ALIASGenerator
from utils import load_checkpoint, save_images

app = Flask(__name__)
app.config['RESULTS_FOLDER'] = 'results'

os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Model options
class ModelOptions:
    def __init__(self):
        self.init_type = 'xavier'
        self.init_variance = 0.02
        self.norm_G = 'spectralaliasinstance'
        self.ngf = 64
        self.num_upsampling_layers = 'most'
        self.semantic_nc = 13
        self.load_height = 1024
        self.load_width = 768
        self.grid_size = 5
        self.dataset_dir = './datasets/'
        self.dataset_mode = 'test'
        self.dataset_list = 'test_pairs.txt'
        self.save_dir = app.config['RESULTS_FOLDER']
        self.name = 'demo'

opt = ModelOptions()

# Load models ONCE
print("Loading models...")
seg_model = SegGenerator(opt, 21, 13)
load_checkpoint(seg_model, './checkpoints/seg_final.pth')
seg_model.cpu().eval()
gmm_model = GMM(opt, 7, 3)
load_checkpoint(gmm_model, './checkpoints/gmm_final.pth')
gmm_model.cpu().eval()
opt.semantic_nc = 7
alias_model = ALIASGenerator(opt, 9)
load_checkpoint(alias_model, './checkpoints/alias_final.pth')
alias_model.cpu().eval()
opt.semantic_nc = 13
print("Models loaded successfully!")

# Load the full test dataset ONCE
print("Loading test dataset into memory...")
dataset = VITONDataset(opt)
print(f"Loaded {len(dataset)} test samples.")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'person_image' not in request.files or 'clothing_image' not in request.files:
            return "Error: Both person and clothing images are required", 400
        
        person_file = request.files['person_image']
        clothing_file = request.files['clothing_image']
        
        if person_file.filename == '' or clothing_file.filename == '':
            return "Error: No files selected", 400
        
        # Extract filenames without extension
        person_filename = os.path.splitext(person_file.filename)[0]
        clothing_filename = os.path.splitext(clothing_file.filename)[0]
        
        # Add .jpg extension
        img_name = f"{person_filename}.jpg"
        c_name = f"{clothing_filename}.jpg"
        
        # Find the index in the loaded dataset
        try:
            idx = dataset.img_names.index(img_name)
            c_idx = dataset.c_names['unpaired'].index(c_name)
        except ValueError:
            return f"Error: Person or clothing image not found in test dataset", 400
        
        # Get the sample (person index, but use requested cloth)
        sample = dataset[idx]
        # Overwrite the cloth and mask with the requested clothing
        sample['cloth']['unpaired'] = dataset[c_idx]['cloth']['unpaired']
        sample['cloth_mask']['unpaired'] = dataset[c_idx]['cloth_mask']['unpaired']
        
        # Run inference (same as AR script, but only for this sample)
        import torch.nn.functional as F
        import kornia
        up = torch.nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear')
        gauss = kornia.filters.GaussianBlur2d((15, 15), (3, 3))
        with torch.no_grad():
            parse_agnostic = sample['parse_agnostic'].unsqueeze(0)
            pose = sample['pose'].unsqueeze(0)
            c = sample['cloth']['unpaired'].unsqueeze(0)
            cm = sample['cloth_mask']['unpaired'].unsqueeze(0)
            img_agnostic = sample['img_agnostic'].unsqueeze(0)
            parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
            pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
            c_masked_down = F.interpolate(c * cm, size=(256, 192), mode='bilinear')
            cm_down = F.interpolate(cm, size=(256, 192), mode='bilinear')
            noise = torch.randn_like(cm_down)
            seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, pose_down, noise), dim=1)
            parse_pred_down = seg_model(seg_input)
            parse_pred = gauss(up(parse_pred_down))
            parse_pred = parse_pred.argmax(dim=1)[:, None]
            parse_old = torch.zeros(parse_pred.size(0), 13, opt.load_height, opt.load_width, dtype=torch.float)
            parse_old.scatter_(1, parse_pred, 1.0)
            labels = {
                0:  ['background',  [0]],
                1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
                2:  ['upper',       [3]],
                3:  ['hair',        [1]],
                4:  ['left_arm',    [5]],
                5:  ['right_arm',   [6]],
                6:  ['noise',       [12]]
            }
            parse = torch.zeros(parse_pred.size(0), 7, opt.load_height, opt.load_width, dtype=torch.float)
            for j in range(len(labels)):
                for label in labels[j][1]:
                    parse[:, j] += parse_old[:, label]
            agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='nearest')
            parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode='nearest')
            pose_gmm = F.interpolate(pose, size=(256, 192), mode='nearest')
            c_gmm = F.interpolate(c, size=(256, 192), mode='nearest')
            gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)
            _, warped_grid = gmm_model(gmm_input, c_gmm)
            warped_c = F.grid_sample(c, warped_grid, padding_mode='border')
            warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')
            misalign_mask = parse[:, 2:3] - warped_cm
            misalign_mask[misalign_mask < 0.0] = 0.0
            parse_div = torch.cat((parse, misalign_mask), dim=1)
            parse_div[:, 2:3] -= misalign_mask
            output = alias_model(torch.cat((img_agnostic, pose, warped_c), dim=1), parse, parse_div, misalign_mask)
            result_name = f"{img_name.split('.')[0]}_{c_name.split('.')[0]}.jpg"
            save_images(output, [result_name], opt.save_dir)
        return send_from_directory(opt.save_dir, result_name)
    except Exception as e:
        print(f"Error in upload: {str(e)}")
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    print('Starting Flask application (optimized, in-memory dataset)...')
    app.run(debug=True) 