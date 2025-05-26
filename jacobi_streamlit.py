import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Kalkulator Metode Jacobi")

st.markdown("### Masukkan Koefisien Sistem Persamaan Linear (Ax = b)")

# Label untuk variabel (X, Y, Z)
var_labels = ['X', 'Y', 'Z']
A = []
B = []

# Input sistem persamaan
for i in range(3):
    st.markdown(f"#### Persamaan {i+1}")
    cols = st.columns(4)
    a_row = [
        cols[j].number_input(f"{var_labels[j]} (Persamaan {i+1})", value=1.0 if i == j else 0.0, key=f"a{i}{j}")
        for j in range(3)
    ]
    b_val = cols[3].number_input(f"B (Persamaan {i+1})", value=0.0, key=f"b{i}")
    A.append(a_row)
    B.append(b_val)

st.markdown("### Masukkan Tebakan Awal")
x0 = st.number_input("Tebakan Awal X₀", value=0.0)
y0 = st.number_input("Tebakan Awal Y₀", value=0.0)
z0 = st.number_input("Tebakan Awal Z₀", value=0.0)

max_iter = st.number_input("Jumlah Iterasi Maksimum", min_value=1, value=25)
tolerance = st.number_input("Toleransi Error", min_value=1e-8, value=1e-6, format="%.8f")

if st.button("Hitung"):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    X = np.array([x0, y0, z0], dtype=float)

    data = []
    errors = []

    # Iterasi 0 (tebakan awal)
    data.append({
        "Iterasi": 0,
        "X": X[0], "Y": X[1], "Z": X[2],
        "Jarak X": None, "Jarak Y": None, "Jarak Z": None, "Error": None
    })

    for i in range(1, max_iter + 1):
        X_new = np.copy(X)

        for j in range(3):
            sum_ = sum(A[j][k] * X[k] for k in range(3) if k != j)
            X_new[j] = (B[j] - sum_) / A[j][j]

        diff = np.abs(X_new - X)
        error = np.max(diff)
        errors.append(error)

        data.append({
            "Iterasi": i,
            "X": X_new[0], "Y": X_new[1], "Z": X_new[2],
            "Jarak X": diff[0], "Jarak Y": diff[1], "Jarak Z": diff[2],
            "Error": error
        })

        X = X_new

        if error < tolerance:
            break

    # Tampilkan tabel hasil iterasi
    df = pd.DataFrame(data)
    st.subheader("Tabel Iterasi Jacobi")
    st.dataframe(df.style.format(precision=6))

    # Tampilkan grafik error vs iterasi
    st.subheader("Grafik Konvergensi Error")
    fig, ax = plt.subplots()
    ax.plot(range(1, len(errors)+1), errors, marker='o', linestyle='-')
    ax.set_xlabel("Iterasi")
    ax.set_ylabel("Error Maksimum")
    ax.set_title("Error vs Iterasi")
    ax.grid(True)
    st.pyplot(fig)
