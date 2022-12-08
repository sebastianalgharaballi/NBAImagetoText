import torch
import torchvision.transforms as transforms
from PIL import Image

# This is a utilities file mainly used to clearly see the names our model is predicting versus the correct names of the test examples through my first function. I also wrote two functions that can be used to save a checkpoint of our current trained model and load that same checkpoint for further training or evaluation.

def print_examples(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)), #resize input image to generic size
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #normalize image in range [-1, 1]
        ]
    )

    model.eval()
    
    test_img1 = transform(Image.open("test_examples/anthony.jpg").convert("RGB")).unsqueeze(0)
    print("Example 1 CORRECT: Anthony Davis")
    print(
        "Example 1 OUTPUT: "
        + " ".join(model.caption_image(test_img1.to(device), dataset.vocab)))
    
    test_img2 = transform(Image.open("test_examples/damian.jpg").convert("RGB")).unsqueeze(0)
    print("Example 2 CORRECT: Damian Lillard")
    print(
        "Example 2 OUTPUT: "
        + " ".join(model.caption_image(test_img2.to(device), dataset.vocab)))
    
    test_img3 = transform(Image.open("test_examples/devin.jpg").convert("RGB")).unsqueeze(0)
    print("Example 3 CORRECT: Devin Booker")
    print(
        "Example 3 OUTPUT: "
        + " ".join(model.caption_image(test_img3.to(device), dataset.vocab)))
    
    test_img4 = transform(Image.open("test_examples/donovan.jpg").convert("RGB")).unsqueeze(0)
    print("Example 4 CORRECT: Donovan Mitchell")
    print(
        "Example 4 OUTPUT: "
        + " ".join(model.caption_image(test_img4.to(device), dataset.vocab)))
    
    test_img5 = transform(Image.open("test_examples/giannis.jpg").convert("RGB")).unsqueeze(0)
    print("Example 5 CORRECT: Giannis Antetokounmpo")
    print(
        "Example 5 OUTPUT: "
        + " ".join(model.caption_image(test_img5.to(device), dataset.vocab)))
    
    test_img6 = transform(Image.open("test_examples/ja.jpg").convert("RGB")).unsqueeze(0)
    print("Example 6 CORRECT: Ja Morant")
    print(
        "Example 6 OUTPUT: "
        + " ".join(model.caption_image(test_img6.to(device), dataset.vocab)))
    
    test_img7 = transform(Image.open("test_examples/jayson.jpg").convert("RGB")).unsqueeze(0)
    print("Example 7 CORRECT: Jayson Tatum")
    print(
        "Example 7 OUTPUT: "
        + " ".join(model.caption_image(test_img7.to(device), dataset.vocab)))
    
    test_img8 = transform(Image.open("test_examples/joel.jpg").convert("RGB")).unsqueeze(0)
    print("Example 8 CORRECT: Joel Embiid")
    print(
        "Example 8 OUTPUT: "
        + " ".join(model.caption_image(test_img8.to(device), dataset.vocab)))
    
    test_img9 = transform(Image.open("test_examples/kawhi.jpg").convert("RGB")).unsqueeze(0)
    print("Example 9 CORRECT: Kawhi Leonard")
    print(
        "Example 9 OUTPUT: "
        + " ".join(model.caption_image(test_img9.to(device), dataset.vocab)))
    
    test_img10 = transform(Image.open("test_examples/kevin.jpg").convert("RGB")).unsqueeze(0)
    print("Example 10 CORRECT: Kevin Durant")
    print(
        "Example 10 OUTPUT: "
        + " ".join(model.caption_image(test_img10.to(device), dataset.vocab)))
    
    test_img11 = transform(Image.open("test_examples/lebron.jpg").convert("RGB")).unsqueeze(0)
    print("Example 11 CORRECT: LeBron James")
    print(
        "Example 11 OUTPUT: "
        + " ".join(model.caption_image(test_img11.to(device), dataset.vocab)))
    
    test_img12 = transform(Image.open("test_examples/luka.jpg").convert("RGB")).unsqueeze(0)
    print("Example 12 CORRECT: Luka Doncic")
    print(
        "Example 12 OUTPUT: "
        + " ".join(model.caption_image(test_img12.to(device), dataset.vocab)))
    
    test_img13 = transform(Image.open("test_examples/nikola.jpg").convert("RGB")).unsqueeze(0)
    print("Example 13 CORRECT: Nikola Jokic")
    print(
        "Example 13 OUTPUT: "
        + " ".join(model.caption_image(test_img13.to(device), dataset.vocab)))
    
    test_img14 = transform(Image.open("test_examples/paul.jpg").convert("RGB")).unsqueeze(0)
    print("Example 14 CORRECT: Paul George")
    print(
        "Example 14 OUTPUT: "
        + " ".join(model.caption_image(test_img14.to(device), dataset.vocab)))
    
    test_img15 = transform(Image.open("test_examples/stephen.jpg").convert("RGB")).unsqueeze(0)
    print("Example 15 CORRECT: Stephen Curry")
    print(
        "Example 15 OUTPUT: "
        + " ".join(model.caption_image(test_img15.to(device), dataset.vocab)))
    
    model.train()


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step
