from Train_evaluate_7887 import *
from modeling_trial3 import *
from torch.optim import AdamW

selected_classes = [
        "bpsk",
        "qpsk",
        "8psk",
        "16qam",
        "64qam",
        "256qam",
        "ofdm-64",
        "ofdm-72",
        "ofdm-128",
        "ofdm-180",
        "ofdm-256",
        "ofdm-300",
        "ofdm-512",
        "ofdm-600",
        "ofdm-900",
        "ofdm-1024",
        "ofdm-1200",
        "ofdm-2048",
    ]

device = torch.device('cuda')
plot_path = "/mnt/beegfs/home/mutaz.abueisheh/torchsig/Gradiant/plots/7887"

transform = ST.Compose([
    ST.Normalize(norm=np.inf),
    ST.ComplexTo2D(),
])
tt = ST.DescToClassIndex(class_list=selected_classes)
root = "/mnt/beegfs/home/mutaz.abueisheh/torchsig/Gradiant/sig18/"
impaired = True
batch_size = 128

train_dataloader, val_dataloader, test_dataloader = prepare_data(
    root, selected_classes, transform, tt, impaired, batch_size
)

model = ResNet1D(Bottleneck1D, [7,8,8,7], num_classes=len(selected_classes), in_channels=2)
model = model.to(device)
num_epochs = 100
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.004)

trainer = prepare_model(model, train_dataloader, val_dataloader, device, criterion, optimizer, num_epochs)

trainer.run_training_loop(num_epochs)

save_metrics_plot(trainer, plot_path)