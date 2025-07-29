# Hyper-Reducing a Residual Matrix with Randomized SVD and the Empirical Cubature Method (ECM)

This guide shows how to apply **Randomized SVD** and the **Empirical Cubature Method (ECM)** to compute a reduced integration rule for a residual matrix of the form:

```
C ∈ ℝ^{ (n_modes × n_snapshots) × n_elements }
```

Each column corresponds to an element or quadrature point, and each row corresponds to the residual of a given snapshot projected onto a POD mode.

---

## 1. Build your training matrix `C`

Ensure your matrix `C` is of shape:
- `n_rows = n_modes × n_snapshots`
- `n_cols = n_elements`

Also compute:

```python
b = C.sum(axis=1)   # Vector used to check the residual norm
```

---

## 2. Extract a residual basis with Randomized SVD

```python
from randomized_singular_value_decomposition import RandomizedSingularValueDecomposition as RSVD

u, _, _, _ = RSVD().Calculate(C.T, truncation_tolerance=1e-8)
```

- The input is `C.T` (shape: `n_elements × (n_modes × n_snapshots)`).
- The output `u` is a matrix of shape `n_elements × r` representing the reduced residual basis.

You can adjust `truncation_tolerance` to control how many basis vectors are kept.

---

## 3. Run the Empirical Cubature Method

```python
from empirical_cubature_method import EmpiricalCubatureMethod as ECM

ecm = ECM(ECM_tolerance=1e-6)
ecm.SetUp(
    ResidualsBasis=u,
    InitialCandidatesSet=None,
    constrain_sum_of_weights=True,
    constrain_conditions=False
)
ecm.Run()
```

- Outputs:
  - `ecm.z` → indices of selected elements
  - `ecm.w` → associated positive weights

---

## 4. Assemble the global weight vector

```python
weights = np.zeros(C.shape[1])
weights[ecm.z] = ecm.w

# Optional: check residual accuracy
error = np.linalg.norm(C @ weights - b) / np.linalg.norm(b)
print(f"ECM residual: {error:.2e}")

# Save the weights
np.save("ecm_weights.npy", weights)
```

---

## 5. Optional Parameters to Adjust

| Parameter                    | Description                                             |
|-----------------------------|---------------------------------------------------------|
| `truncation_tolerance`      | Tolerance passed to RSVD; controls basis size `r`       |
| `ECM_tolerance`             | Target relative error for the cubature method           |
| `constrain_sum_of_weights`  | Enforces ∑ w ≈ n_elements to avoid trivial solutions     |
| `constrain_conditions`      | If True, treats the last elements as boundary conditions |
| `number_of_conditions`      | Number of trailing elements considered boundary-related  |

---

## Output

- `ecm_weights.npy`: array of weights for the selected elements, with zeros elsewhere.
- Can be passed directly to your hyper-reduced online solver.
