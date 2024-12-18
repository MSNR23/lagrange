import sympy as sp
from concurrent.futures import ProcessPoolExecutor
import time

def load_energies(filepath):
    """Load potential and kinetic energy from a text file."""
    with open(filepath, "r") as file:
        content = file.read()

    print("=== Loaded File Content ===")
    print(content)  # デバッグ用出力

    # Define symbolic variables used in the energies
    t = sp.symbols('t')
    g, l1, lg1, lg2, m1, m2, I1, Iyy2 = sp.symbols('g l1 lg1 lg2 m1 m2 I1 Iyy2')
    theta2 = sp.Function('theta2')(t)
    theta2_dot = sp.diff(theta2, t)
    q = [sp.Function(f'q{i}')(t) for i in ['0', '1', '2', '3']]
    q_dot = [sp.diff(qi, t) for qi in q]

    # Replace placeholders in the text with sympy symbols
    replacements = {
        "q10(t)": str(q[0]),
        "q11(t)": str(q[1]),
        "q12(t)": str(q[2]),
        "q13(t)": str(q[3]),
        "q10_dot": str(q_dot[0]),
        "q11_dot": str(q_dot[1]),
        "q12_dot": str(q_dot[2]),
        "q13_dot": str(q_dot[3]),
        "theta2_dot": str(theta2_dot),
    }

    for placeholder, symbol in replacements.items():
        if placeholder in content:
            content = content.replace(placeholder, symbol)
        else:
            print(f"Warning: Placeholder {placeholder} not found in file.")

    # Extract energies
    try:
        potential_energy = content.split("Potential Energy:\n")[1].split("\n\n")[0]
        translational_kinetic_energy = content.split("Translational Kinetic Energy:\n")[1].split("\n\n")[0]
        rotational_kinetic_energy = content.split("Rotational Kinetic Energy:\n")[1]
    except IndexError as e:
        raise ValueError("The energy file format is incorrect or incomplete.") from e

    print("=== Parsed Energies ===")
    print("Potential Energy (U):", potential_energy)
    print("Translational Kinetic Energy (T_trans):", translational_kinetic_energy)
    print("Rotational Kinetic Energy (T_rot):", rotational_kinetic_energy)

    # Convert strings to SymPy expressions
    U = sp.sympify(potential_energy)
    T_trans = sp.sympify(translational_kinetic_energy)
    T_rot = sp.sympify(rotational_kinetic_energy)

    return T_trans, T_rot, U, q, q_dot, theta2

def compute_lagrange_equation(T, U, q, dq, t):
    """Compute Lagrange equation for a single variable."""
    L = T - U
    dL_dq = sp.diff(L, q)
    dL_ddq = sp.diff(L, dq)
    ddt_dL_ddq = sp.diff(dL_ddq, t)
    return sp.simplify(ddt_dL_ddq - dL_dq)

def save_equation_to_file(equation, filename):
    """Save a single Lagrange equation to a text file."""
    with open(filename, "w") as file:
        file.write(str(equation))

def process_variable(args):
    """Compute and save Lagrange equation for a single variable (parallelized)."""
    T, U, q, dq, t, index = args
    print(f"Computing Lagrange equation for variable {q}...")
    start_time = time.time()
    lagrange_eq = compute_lagrange_equation(T, U, q, dq, t)
    filename = f"lagrange_equation_{index}_{q.func.__name__}.txt"
    save_equation_to_file(lagrange_eq, filename)
    elapsed_time = time.time() - start_time
    print(f"Finished computing {q}. Saved to {filename}. Time taken: {elapsed_time:.2f} seconds.")

def main():
    # Define symbolic parameters
    t = sp.symbols('t')

    # Load energies from file
    filepath = "energies_simplified.txt"
    T_trans, T_rot, U, q_vars, dq_vars, theta2 = load_energies(filepath)
    T_total = T_trans + T_rot

    # Prepare arguments for parallel computation
    args = [(T_total, U, q, dq, t, i) for i, (q, dq) in enumerate(zip(q_vars, dq_vars))]

    # Parallel processing
    with ProcessPoolExecutor() as executor:
        executor.map(process_variable, args)

if __name__ == "__main__":
    main()
