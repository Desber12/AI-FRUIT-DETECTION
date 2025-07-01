from flask import Flask, render_template, request
import torch
from torchvision import models, transforms
from PIL import Image
import os
import json

# ⬇️ Pastikan folder static tersedia
os.makedirs('static', exist_ok=True)

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
with open("classes.json") as f:
    classes = json.load(f)

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("fruit_model.pth", map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img = Image.open(file.stream).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                prob = torch.nn.functional.softmax(output[0], dim=0)
                conf, pred = torch.max(prob, 0)
                result = {
                    'label': classes[pred.item()],
                    'confidence': f"{conf.item()*100:.2f}%",
                    'image': file.filename
                }
                img.save(os.path.join('static', 'uploaded.jpg'))
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
