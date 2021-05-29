import numpy as np
import tensorflow as tf

def complexity(model, ds):

  @tf.function
  def predict(x):
    logits = model(x)
    pred = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
    return pred

  def estimate_accuracy(model, dataset, num_iter=500):
    acc = 0.0
    for i in range(num_iter):
      x, y = dataset.next()
      pred = predict(x).numpy()
      acc += np.mean(pred == y)
    return acc / num_iter

  def regularizedW(W, n_comp=None):
    N, M = np.max(W.shape), np.min(W.shape)

    if n_comp is None:
      n_comp = int(M*0.2)

    u, s, vh = np.linalg.svd(W, compute_uv=True)
    # regularize
    s[n_comp:]=0

    s = list(s)
    s.extend([0]*(N-M))
    s = np.array(s)
    s = np.diag(s)
    if u.shape[0] > vh.shape[0]:
      reg_W = np.dot(np.dot(u,s)[:N,:M],vh)
    else:
      reg_W = np.dot(u, np.dot(s,vh)[:M,:N])

    return reg_W


  batched_ds = iter(ds.shuffle(1000).repeat(-1).batch(64))

  data_pair = batched_ds.next()
  #print(model.summary())

  #for il, l in enumerate(model.layers):
    #if len(l.trainable_weights) > 0:
      #print(il, l.trainable_weights[0].shape)

  #print("--------")

  new_weights = []
  for iw,w in enumerate(model.get_weights()):
    new_w = w # reset below if possible

    s = w.shape
    #print(iw, s)
    if len(s)==4:
      k = s[0]
      N = np.max([s[2:]])
      M = np.min(s[2:])
      if M > 10 and N < 10000:
        #print("len 4 {} {}".format(M,N))
        for i in range(k):
          for j in range(k):
            w_slice = w[i,j,:]
            new_w[i,j,:] = regularizedW(w_slice)

    elif len(s)==2:
      N = np.max(s)
      M = np.min(s)
      if M > 10 and N < 10000:
        #print("len 2 {} {}".format(M,N))
        new_w = regularizedW(w)


    new_weights.append(tf.convert_to_tensor(new_w,  dtype=tf.float32))

  model.set_weights(new_weights)
  predicted_accuracy = estimate_accuracy(model, batched_ds)
  #print(predicted_accuracy)
  return predicted_accuracy

