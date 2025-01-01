def evaluate_model(model, test_data, test_labels):
    # Evaluate the model on the test dataset
    loss, accuracy = model.evaluate(test_data, test_labels)
    return loss, accuracy

def generate_report(loss, accuracy):
    # Generate a report of the evaluation results
    report = {
        'Loss': loss,
        'Accuracy': accuracy
    }
    return report