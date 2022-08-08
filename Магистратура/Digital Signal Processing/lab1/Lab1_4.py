import numpy.random as rand
import math
from matplotlib import pyplot as plt
import numpy as np


#* Sampling interval
T = 0.2

def get_omega_matrix(w):
  return(np.array([
                [0, -w[0], -w[1], -w[2]],
                  [w[0], 0, w[2], -w[1]],
                  [w[1], -w[2], 0, w[0]],
                  [w[2], w[1], -w[0], 0]]))

def get_ks_matrix(q):
  return(np.array([
                   [-q[1], -q[2], -q[3]],
                   [q[0], -q[3], q[2]],
                   [q[3], q[0], -q[1]],
                   [-q[2], q[1], q[0]]])
                   )

#* Rotation matrix for q
def Q(q):
  return(np.array([
    [2 * q[0] ** 2 - 1 + 2 * q[1] ** 2, 2 * q[1] * q[2] - 2 * q[0] * q[3], 2 * q[1] * q[3] + 2 * q[0] * q[2]],
    [2 * q[1] * q[2] + 2 * q[0] * q[3], 2 * q[0] ** 2 - 1 + 2 * q[2] ** 2, 2 * q[2] * q[3] - 2 * q[0] * q[1]],
    [2 * q[1] * q[3] - 2 * q[0] * q[2], 2 * q[2] * q[3] + 2 * q[0] * q[1], 2 * q[0] ** 2 - 1 + 2 * q[3] ** 2]]))

#* Jacobi matrix for Q
def Q_J(q, g):
    return(np.array([
        np.array([
                        [4 * q[0], 2 * q[3], -2 * q[2]],
                        [-2 * q[3], 4 * q[0], 2 * q[1]],
                        [2 * q[2], -2 * q[1], 4 * q[0]]]) @ g
        , np.array([
                        [4 * q[1], 2 * q[2], 2 * q[3]],
                        [2 * q[2], 0, 2 * q[0]],
                        [2 * q[3], -2 * q[0], 0]]) @ g
        , np.array([
                        [0, 2 * q[1], -2 * q[0]],
                        [2 * q[1], 4 * q[2], 2 * q[3]],
                        [2 * q[0], 2 * q[3], 0]]) @ g
        , np.array([
                        [0, 2 * q[0], 2 * q[1]],
                        [-2 * q[0], 0, 2 * q[2]],
                        [2 * q[1], 2 * q[2], 4 * q[3]]]) @ g
            ]).T.reshape(3,4)
        )

def A(w):
  return(np.eye(4, 4) + 0.5 * get_omega_matrix(w) * T)

def B(q):
  return(T / 2 * get_ks_matrix(q))

def F_x(w):
  return(A(w))

def F_q(q):
  return(B(q))

#*Dynamic of orientation
def get_q(q, w, Q):
  return(A(w) @ q) #+ B(q) @ np.random.multivariate_normal(mean=np.zeros(3), cov=Q))

def get_y(q, g, R):
  return(Q(q).T @ g + np.random.multivariate_normal(mean=np.zeros(3), cov=R).reshape(-1, 1))

def quat_normalize(q):
  s = np.sign(q[0])
  quat_norm = np.linalg.norm(q)
  return(np.array([s * q[i] / quat_norm for i in range(4)]))

def main():
    images_root = 'imgs/{}'
    iters_num = 120

    sigma = 0.021
    mu = 0.04
    p_sigma = 0.01
    g = np.array([0,0,9.81]).reshape(3, 1)

    #* Cov matrix for moving noize
    Q_values = np.eye(3, 3) * pow(mu, 2) 
    
    #* Cov matrix for obs
    R_obs = np.eye(3, 3) * pow(sigma, 2) 

    #* Initial values
    q_i = np.array([0, 1, 0, 0]) #* orient
    y_i = np.array([0, 0, 0]) #* acc
    m_i = np.array([0, 1, 0, 0]) 
    w_i = np.array([1, 1, 0]) 
    p_i = np.eye(4, 4) * pow(p_sigma,  2)

    true_values = [list([q_i[i]]) for i in range(4)] 
    filtred_values = [list([m_i[i]]) for i in range(4)]
    last_m_i = m_i

    for i in range(iters_num):
        y_i = get_y(q_i, g, R_obs)
        q_i = quat_normalize(get_q(q_i, w_i, Q_values))

        for j in range(4):
            true_values[j].append(q_i[j])

        #* Filtration

        # Predict
        m_i_predicted = get_q(m_i, w_i, Q_values)
        p_i_predicted = F_x(w_i) @ p_i @ F_x(w_i).T + F_q(m_i) @ Q_values @ F_q(m_i).T
        
        # Correction
        S_i = Q_J(m_i_predicted, g) @ p_i_predicted @ Q_J(m_i_predicted,g).T + Q_values
        K_i = p_i_predicted @ Q_J(m_i_predicted,g).T @ np.linalg.inv(S_i)
        m_i = quat_normalize((m_i_predicted.reshape(-1, 1) + K_i @ (y_i - get_y(m_i_predicted,g, R_obs))).T[0])
        #* Update
        p_i = p_i_predicted - K_i @ S_i @ K_i.T

        for j in range(4):
            filtred_values[j].append(m_i[j])
        break

    for j in range(4):
        plt.plot(true_values[j])
        plt.plot(filtred_values[j])
        plt.legend(["True", "Filtred"])
        plt.savefig(images_root.format(f'Lab1_4_orient_{j+1}.png'))
        plt.close()


if __name__ == '__main__':
    main()