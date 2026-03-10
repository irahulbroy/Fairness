import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import streamlit as st
import io
import qrcode

# -----------------------------
# Fixed Parameters
# -----------------------------
T = 100
k1 = 0.0065
k2 = 0.0065
p1 = np.full(T, 0.9)
p2 = np.full(T, 0.1)
alpha_grid = np.linspace(0, 50, 100)

st.set_page_config(layout="wide")
st.title("Fairness as an Asset – Interactive Model")

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Model Parameters")
beta = st.sidebar.slider("Beta (β)", 0.2, 1.0, 0.3, 0.01)
gamma1 = st.sidebar.slider("Gamma 1 (γ₁)", 0.1, 0.9, 0.2, 0.01)
gamma2 = st.sidebar.slider("Gamma 2 (γ₂)", 0.1, 0.9, 0.4, 0.01)
u1 = st.sidebar.slider("u₁", 0.1, 1.0, 1.0, 0.1)
u2 = st.sidebar.slider("u₂", 0.1, 1.0, 0.8, 0.1)

# -----------------------------
# Core Model
# -----------------------------
@st.cache_data
def compute_model(beta, gamma1, gamma2, u1, u2):
    V_vals, Total_provider_vals, Total_surplus_vals = [], [], []
    LHS_vals, RHS_vals = [], []
    for alpha in alpha_grid:
        phi1 = np.exp(-k1 * alpha)
        phi2 = np.exp(-k2 * alpha)
        r1 = gamma1 * u1 * phi1
        r2 = gamma2 * u2 * phi2
        w1_coeff = (1 - gamma1) * u1 * phi1
        w2_coeff = (1 - gamma2) * u2 * phi2
        A = (r1 + beta * r2) * p1
        B = (r2 + beta * r1) * p2
        c = -(A - B)
        A_ub = np.array([[2.0] * T, [-2.0] * T])
        b_ub = np.array([alpha + T, alpha - T])
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0,1)]*T, method='highs')
        if not res.success:
            V_vals.append(np.nan)
            Total_provider_vals.append(np.nan)
            Total_surplus_vals.append(np.nan)
            LHS_vals.append(np.nan)
            RHS_vals.append(np.nan)
            continue
        x_star = res.x
        V = np.sum(A * x_star + B * (1 - x_star))
        W1 = np.sum(w1_coeff * (p1 * x_star + beta * p2 * (1 - x_star)))
        W2 = np.sum(w2_coeff * (p2 * (1 - x_star) + beta * p1 * x_star))
        Total_provider = W1 + W2
        Total_surplus = V + Total_provider
        R1_star = np.sum(r1 * (p1 * x_star + beta * p2 * (1 - x_star)))
        R2_star = np.sum(r2 * (p2 * (1 - x_star) + beta * p1 * x_star))
        LHS = k1 * R1_star + k2 * R2_star
        shadow_price = np.sum(np.abs(res.ineqlin.marginals))
        V_vals.append(V)
        Total_provider_vals.append(Total_provider)
        Total_surplus_vals.append(Total_surplus)
        LHS_vals.append(LHS)
        RHS_vals.append(shadow_price)
    return V_vals, Total_provider_vals, Total_surplus_vals, LHS_vals, RHS_vals

# Compute
V_vals, Total_provider_vals, Total_surplus_vals, LHS_vals, RHS_vals = compute_model(beta, gamma1, gamma2, u1, u2)

# -----------------------------
# Layout using Streamlit columns (2x2)
# -----------------------------
cols = st.columns(2)

# --- Top Left: Platform Revenue V(α) ---
with cols[0]:
    fig1, ax1 = plt.subplots(figsize=(8,6))
    ax1.plot(alpha_grid[:len(V_vals)], V_vals, linewidth=2)
    #ax1.set_title(r"Platform's Optimal Revenue vs. $\alpha$")
    ax1.set_xlabel(r"$\alpha$")
    ax1.set_ylabel(r"$V(\alpha)$")
    ax1.grid(False)
    optimal_idx = np.nanargmax(V_vals)
    ax1.axvline(alpha_grid[optimal_idx], color='red', linestyle='--', label=fr"Optimal $\alpha$: {alpha_grid[optimal_idx]:.2f}")
    ax1.legend()
    st.pyplot(fig1)
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format="png", dpi=300, bbox_inches="tight")
    st.download_button("Download V(α) Plot", buf1.getvalue(), "platform_revenue.png", "image/png")


# --- Top Right: Theorem Test LHS(α) vs RHS(α) ---
with cols[1]:
    fig2, ax2 = plt.subplots(figsize=(8,6))
    ax2.plot(alpha_grid[:len(LHS_vals)], LHS_vals, label=r"LHS($\alpha$) – Retention Benefit")
    ax2.plot(alpha_grid[:len(RHS_vals)], RHS_vals, label=r"RHS($\alpha$) – Value of Unfairness")
    #ax2.set_title("Theorem 1 (Fairness as an Asset)")
    ax2.set_xlabel(r"$\alpha$")
    ax2.set_ylabel("Marginal Values")
    ax2.legend()
    ax2.grid(False)
    optimal_idx = np.nanargmax(V_vals)
    ax2.axvline(alpha_grid[optimal_idx], color='red', linestyle='--', label=fr"Max $V(\alpha)$")
    ax2.legend()
    st.pyplot(fig2)
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", dpi=300, bbox_inches="tight")
    st.download_button("Download LHS/RHS Plot", buf2.getvalue(), "theorem_plot.png", "image/png")

# --- Bottom Left: Social Surplus V(α) + W1*(α) + W2*(α) ---
cols2 = st.columns(2)
with cols2[0]:
    fig3, ax3 = plt.subplots(figsize=(8,6))
    ax3.plot(alpha_grid[:len(Total_surplus_vals)], Total_surplus_vals, color='black', linewidth=2)
    #ax3.set_title(r"Social Surplus vs. $\alpha$")
    ax3.set_xlabel(r"$\alpha$")
    ax3.set_ylabel(r"$V(\alpha) + W_1^*(\alpha) + W_2^*(\alpha)$")
    ax3.grid(False)
    optimal_idx = np.nanargmax(V_vals)
    optimal_surplus_idx = np.nanargmax(Total_surplus_vals)
    ax3.axvline(alpha_grid[optimal_idx], color='red', linestyle='--', label=fr"Platform's Optimal $\alpha$: {alpha_grid[optimal_idx]:.2f}")
    ax3.axvline(alpha_grid[optimal_surplus_idx], color='black', linestyle='--', label=fr"Social Planner's Optimal $\alpha$: {alpha_grid[optimal_surplus_idx]:.2f}")
    ax3.legend()
    st.pyplot(fig3)
    buf3 = io.BytesIO()
    fig3.savefig(buf3, format="png", dpi=300, bbox_inches="tight")
    st.download_button("Download Social Surplus", buf3.getvalue(), "social_surplus.png", "image/png")

# --- Bottom Right: Total Provider Surplus W1*(α) + W2*(α) ---
with cols2[1]:
    fig4, ax4 = plt.subplots(figsize=(8,6))
    ax4.plot(alpha_grid[:len(Total_provider_vals)], Total_provider_vals, color='magenta', linewidth=2)
    #ax4.set_title("Total Provider Surplus vs. $\alpha$")
    ax4.set_xlabel("$\alpha$")
    ax4.set_ylabel(r"$W(\alpha) = W_1^*(\alpha) + W_2^*(\alpha)$")
    ax4.grid(False)
    optimal_idx = np.nanargmax(V_vals)
    optimal_earning_idx = np.nanargmax(Total_provider_vals)
    ax4.axvline(alpha_grid[optimal_idx], color='red', linestyle='--', label=fr"Platform's Optimal $\alpha$: {alpha_grid[optimal_idx]:.2f}")
    ax4.axvline(alpha_grid[optimal_earning_idx], color='black', linestyle='--', label=fr"Total Provider Optimal $\alpha$: {alpha_grid[optimal_earning_idx]:.2f}")
    ax4.legend()
    st.pyplot(fig4)
    buf4 = io.BytesIO()
    fig4.savefig(buf4, format="png", dpi=300, bbox_inches="tight")
    st.download_button("Download Provider Surplus", buf4.getvalue(), "provider_surplus.png", "image/png")


# -----------------------------
# QR Code Section
# -----------------------------
st.markdown("---")
st.subheader("Share This App")
app_url = st.text_input("Paste your deployed app URL here to generate QR code")

if app_url:
    # Create QR code as PIL image
    qr = qrcode.make(app_url)
    
    # Convert PIL image to bytes for Streamlit
    buf_qr = io.BytesIO()
    qr.save(buf_qr, format="PNG")
    buf_qr.seek(0)  # Important: move cursor to start
    
    st.image(buf_qr, caption="Scan to open app", use_column_width=True)
    
    
    # ---------------- Footer / Copyright ---------------- #
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 14px;'>
        &copy; 2026 Rahul Roy. All Rights Reserved.<br>
        <i>This educational simulation may not be reproduced or distributed without explicit permission.</i>
    </div>
    """, 
    unsafe_allow_html=True
)