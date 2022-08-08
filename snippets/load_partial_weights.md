Load weights from pretrained models when you have modified the model architecture.



This function only load weights that match in names and shapes.
```python
def load_partial_weights(model, state_dict):
    '''
    args:
        model: model to load weights into
        state_dict: state_dict to load weights from
    '''
    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    print(f"{len(filtered_dict)} from {len(state_dict)} weights loaded")
```

