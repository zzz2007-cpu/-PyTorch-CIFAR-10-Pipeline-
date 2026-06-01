import torch
from torchvision import datasets, transforms


# CIFAR-10 three-channel mean and standard deviation.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_transforms(data_aug: bool):
    if data_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    return train_transform, test_transform


def apply_label_noise(dataset, noise_rate: float, seed: int, num_classes: int = 10):
    if noise_rate <= 0:
        return 0

    if noise_rate >= 1:
        raise ValueError("label_noise_rate should be in [0, 1).")

    generator = torch.Generator()
    generator.manual_seed(seed)

    targets = torch.tensor(dataset.targets, dtype=torch.long)
    num_samples = len(targets)
    num_noisy = int(num_samples * noise_rate)

    noisy_indices = torch.randperm(num_samples, generator=generator)[:num_noisy]

    old_labels = targets[noisy_indices]

    # Generate wrong labels. This avoids replacing a label with itself.
    random_offsets = torch.randint(
        low=1,
        high=num_classes,
        size=(num_noisy,),
        generator=generator,
    )
    new_labels = (old_labels + random_offsets) % num_classes

    targets[noisy_indices] = new_labels
    dataset.targets = targets.tolist()

    return num_noisy


def build_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    data_aug: bool,
    label_noise_rate: float = 0.0,
    label_noise_seed: int = 42,
):
    train_transform, test_transform = build_transforms(data_aug)

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    num_noisy = apply_label_noise(
        dataset=train_dataset,
        noise_rate=label_noise_rate,
        seed=label_noise_seed,
        num_classes=len(train_dataset.classes),
    )

    if num_noisy > 0:
        print(
            f"Applied label noise: {num_noisy}/{len(train_dataset)} "
            f"({label_noise_rate:.1%}) training labels changed."
        )

    pin_memory = torch.cuda.is_available()

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader, train_dataset.classes
