import re
import matplotlib.pyplot as plt

def parse_log_file(file_path):
    epochs = []
    train_losses = []
    test_losses = []
    
    with open(file_path, 'r') as file:
        content = file.read()
        
    train_pattern = r"epoch = (\d+), training loss = ([\d.]+)"
    test_pattern = r"test accuracy = [\d.]+±[\d.]+, test loss = ([\d.]+)"
    
    train_matches = re.findall(train_pattern, content)
    test_matches = re.findall(test_pattern, content)
    
    for train_match, test_match in zip(train_matches, test_matches):
        epoch = int(train_match[0])
        train_loss = float(train_match[1])
        test_loss = float(test_match)
        
        epochs.append(epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    
    return epochs, train_losses, test_losses

def plot_losses(epochs, train_losses, test_losses):
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.title('Training and Test Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.show()

# Replace 'log_file.txt' with the path to your log file
file_path = 'run_train32.out'

epochs, train_losses, test_losses = parse_log_file(file_path)
plot_losses(epochs, train_losses, test_losses)

# Print some statistics
print(f"Total epochs: {len(epochs)}")
print(f"Initial training loss: {train_losses[0]}")
print(f"Final training loss: {train_losses[-1]}")
print(f"Initial test loss: {test_losses[0]}")
print(f"Final test loss: {test_losses[-1]}")