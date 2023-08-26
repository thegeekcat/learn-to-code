# Define a function: Calculate predictive accuracy of models
def get_acc(model, data_loader, devices):
    # Initialize parameters
    correct_pred = 0
    n = 0
    
    model.eval()
    
    with torch.no_grad():
        for x, y_true in data_loader:
            x = x.to(device)
            y - y_true.to(device)
            
            _y, y_prob = model(x)
            _, predicted_labels = torch.max(y_porbl, 1)
            
            n += y.size(0)
            conrrect_pred += (predicted_labels == y).sum()
    
    return correct_pred.float() / n