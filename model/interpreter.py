def feature_importance_knockout(model,src, frac, method='zero'):
  model.eval()
  output = model(src, frac)
  len_feat = len(src[0][0])
  for n in range(len(feat)):
    if method=='zero':
      subs_vector = torch.zeros(len_feat)
