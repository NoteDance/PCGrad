# PCGrad

**Overview**:

The `PCGrad` (Projected Conflicting Gradients) method is a gradient surgery technique designed for multi-task learning. It addresses the issue of conflicting gradients—where gradients from different tasks point in opposing directions—by projecting each task’s gradient onto the normal plane of any other task’s gradient with which it conflicts. This projection mitigates gradient interference, leading to more stable and efficient multi-task optimization.

**Parameters**:

- **`reduction`** *(str, default="mean")*: Method to merge non-conflicting gradients. Options:
  - **`"mean"`**: Averages the shared gradient components.
  - **`"sum"`**: Sums the shared gradient components.

**Key Methods**:

- **`pack_grad(tape, losses, variables)`**: Computes and flattens gradients for each task loss.
  - **`tape`**: A `tf.GradientTape` instance (persistent if reused).
  - **`losses`**: List of loss tensors for each task.
  - **`variables`**: List of model variables.
  - **Returns**: Tuple `(grads_list, shapes, has_grads_list)` where:
    - `grads_list` is a list of flattened gradients per task.
    - `shapes` records original variable shapes for unflattening.
    - `has_grads_list` indicates presence of gradients (mask).

- **`project_conflicting(grads, has_grads)`**: Applies gradient surgery across tasks.
  - **`grads`**: List of flattened task gradients.
  - **`has_grads`**: List of masks for existing gradients.
  - **Returns**: A single merged flattened gradient after resolving conflicts.

- **`pc_backward(tape, losses, variables)`**: End-to-end API to compute PCGrad-adjusted gradients.
  - **Returns**: List of unflattened gradients matching `variables`.

---

**Example Usage**:
```python
import tensorflow as tf

# Define model and tasks
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10)
])

# Instantiate PCGrad
pcgrad = PCGrad(reduction='mean')
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# Custom training step with PCGrad
@tf.function
def train_step(x_batch, y_batch_tasks):
    # y_batch_tasks is a list of labels per task
    with tf.GradientTape(persistent=True) as tape:
        losses = [
            tf.keras.losses.sparse_categorical_crossentropy(y, model(x_batch), from_logits=True)
            for y in y_batch_tasks
        ]
    # Compute PCGrad-adjusted gradients
    pc_grads = pcgrad.pc_backward(tape, losses, model.trainable_variables)
    # Apply gradients
    optimizer.apply_gradients(zip(pc_grads, model.trainable_variables))

# Example training loop
for epoch in range(10):
    for x_batch, y_batch_tasks in train_dataset:
        train_step(x_batch, y_batch_tasks)
```

# PPCGrad

**Overview**:

The `PPCGrad` (Parallel Projected Conflicting Gradients) optimizer extends the PCGrad method by leveraging multiprocessing to parallelize the gradient surgery step. PPCGrad identifies and resolves conflicts among task-specific gradients by projecting each gradient onto the normal plane of any other conflicting gradient, similar to PCGrad. However, PPCGrad distributes the projection computation across multiple processes, which can accelerate the gradient adjustment in multi-core environments, especially when dealing with large models or many tasks.

**Parameters**:

- **`reduction`** *(str, default="mean")*: Method to merge non-conflicting gradient components:
  - **`"mean"`**: Averages the shared gradient components across tasks.
  - **`"sum"`**: Sums the shared gradient components across tasks.

**Key Methods**:

- **`pack_grad(tape, losses, variables)`**: Computes and flattens gradients for each task loss.
  - **`tape`**: A `tf.GradientTape` instance (persistent if reused for multiple losses).
  - **`losses`**: List of loss tensors for each task.
  - **`variables`**: List of model variables.
  - **Returns**: `(grads_list, shapes, has_grads_list)`:
    - `grads_list`: Flattened gradients per task.
    - `shapes`: Original shapes for unflattening.
    - `has_grads_list`: Masks indicating presence of gradients.

- **`project_conflicting(grads, has_grads)`**: Parallel gradient surgery using multiprocessing.
  - **`grads`**: List of flattened task gradients.
  - **`has_grads`**: List of masks for existing gradients.
  - **Returns**: Merged flattened gradient after conflict resolution.

- **`pc_backward(tape, losses, variables)`**: End-to-end API to compute PPCGrad-adjusted gradients.
  - **Returns**: List of unflattened gradients matching `variables`.

---

**Example Usage**:
```python
import tensorflow as tf

# Define model and tasks\ nmodel = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10)
])

# Instantiate PPCGrad
ppcgrad = PPCGrad(reduction='mean')
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# Custom training step with PPCGrad
@tf.function
def train_step(x_batch, y_batch_tasks):
    with tf.GradientTape(persistent=True) as tape:
        losses = [
            tf.keras.losses.sparse_categorical_crossentropy(y, model(x_batch), from_logits=True)
            for y in y_batch_tasks
        ]
    # Compute PPCGrad-adjusted gradients
    ppc_grads = ppcgrad.pc_backward(tape, losses, model.trainable_variables)
    # Apply gradients
    optimizer.apply_gradients(zip(ppc_grads, model.trainable_variables))

# Example training loop\ nfor epoch in range(10):
    for x_batch, y_batch_tasks in train_dataset:
        train_step(x_batch, y_batch_tasks)
```
