import torch

def feature_importance(model,src, frac, method='zero'):
  model.eval()
  if method=='zero':
      baseline value = 0.0
  with torch.no_grad():
      # Get original output from the full model
      original_output = model(src, frac)
      
      # Save the original cbfv embedding weights
      original_weights = model.encoder.embed.cbfv.weight.data.clone()
      feat_size = original_weights.shape[1]
      importance = torch.zeros(feat_size, device=src.device)
      
      # Loop over each feature dimension
      for i in range(feat_size):
          # Clone the original weights and zero out the i-th feature column
          modified_weights = original_weights.clone()
          modified_weights[:, i] = baseline_value
          
          # Temporarily update the embedding weights with the modified version
          model.encoder.embed.cbfv.weight.data.copy_(modified_weights)
          
          # Run a fut_output = model(src, frac)
          
          # Compute a difference metric (e.g., mean absolute difference) between the outputs
          diff = torch.abs(original_output - knockout_output).mean()
          importance[i] = diff
      
      # Restore the original cbfv embedding weights
      model.encoder.embed.cbfv.weight.data.copy_(original_weights)
      
  return importance
