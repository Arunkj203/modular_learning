import torch

def evaluate_lora(model, dataloader, device):
    """
    Evaluates a LoRA-adapted model on the provided dataloader.

    Args:
        model: The LoRA-adapted model to evaluate.
        dataloader: DataLoader providing evaluation data.
        device: Device to run evaluation on ('cpu' or 'cuda').

    Returns:
        float: Average loss over the evaluation dataset.
    """

    model.eval()
    model.to(device)
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    return avg_loss