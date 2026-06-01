import torch
from torchvision import datasets, transforms


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


def apply_label_noise(dataset, noise_rate: float, seed: int):
    if not 0.0 <= noise_rate <= 1.0:
        raise ValueError(f"label_noise_rate must be in [0, 1], got {noise_rate}")

    num_samples = len(dataset.targets)
    num_noisy = int(num_samples * noise_rate)
    if num_noisy == 0:
        return dataset

    generator = torch.Generator().manual_seed(seed)
    noisy_indices = torch.randperm(num_samples, generator=generator)[:num_noisy]
    targets = torch.tensor(dataset.targets, dtype=torch.long)
    num_classes = len(dataset.classes)

    replacement = torch.randint(
        low=0,
        high=num_classes - 1,
        size=(num_noisy,),
        generator=generator,
    )
    original = targets[noisy_indices]
    replacement += (replacement >= original).long()
    targets[noisy_indices] = replacement

    dataset.targets = targets.tolist()
    return dataset


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
    apply_label_noise(train_dataset, label_noise_rate, label_noise_seed)

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
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
