import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
      scores = X[i].dot(W)
      scores -= np.max(scores) # for numerical stability
      exp_sum = np.sum(np.exp(scores))
      loss -= np.log(np.exp(scores[y[i]]) / exp_sum)
      for j in xrange(num_classes):
          # d/dW[j] loss[i] = - X[i] * (1(j = y[i]) - p(y[i] = j | X[i]; W))
          #            = X[i] * p(y[i] = j | X[i]; W) - (1(j = y[i]) * X[i])
          dW[:, j] += np.exp(scores[j]) / exp_sum  * X[i]
          if j == y[i]:
              # j == y[i] ==> 1(j = y[i]) == 1
              dW[:, j] -= X[i]

  loss /= num_train
  dW /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]

  # scores[i] = X[i] * W
  scores = np.dot(X, W)

  scores -= np.max(scores, axis=1).reshape(-1, 1) # for numerical stability
  # exp_sum[i] = sum_{j} (exp(x_i * W_j))
  exp_sum = np.sum(np.exp(scores), axis=1)
  # loss = sum_{i} (-log(exp(scores[y[i]]) / exp_sum[i]))
  loss = np.sum(-1 * np.log(np.exp(scores[np.arange(num_train), y]) / exp_sum))

  # d/dW loss
  P = np.exp(scores)/exp_sum.reshape(-1, 1) # N x C
  P[np.arange(num_train), y] -= 1 # P[i, y[i]] = prob(y[i] == y[i]|x[i];W) - 1
  dW = np.dot(X.T, P)

  loss /= num_train
  dW /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
