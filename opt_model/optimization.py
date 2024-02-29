import numpy as np
import cyipopt
import cvxpy as cp
from cyipopt import minimize_ipopt

## See Test File & Documentation for explanation of class methods
#           as well as solver options
# https://cyipopt.readthedocs.io/en/latest/tutorial.html#tutorial

from jax import jit, grad, jacfwd, jacrev
import jax.numpy as jnp
from cyipopt import minimize_ipopt
import numpy as np

### NOTE {TEST Vars}
TEST = 3
SOLVE = True
PLOT = True

### NOTE {File Description}
''' Collection of various optimization formulations of
    the protocol/speculator interaction problem. The most useful
    classes as of now are :
    - Speculator IP
    - ProtoclLQ  Problem

    Running this file will test the speculator class. Based on test vars above.
'''
### NOTE {Speculator Description}:
''' Speculator plays a follow-the-leader game (sort of) where a desired burn/mint 
    trajectory is given. The speculator then must decide how they would like rate changes to be 
    allocated across the horizon "T" to ensure their best expected return.
    Their constraints are as follows: 
    (1) They must adhere to terminal conditions, which is a predetermined final rate 
    (2) Also, they must obey liquidation risk which is encoded in log penalty function
            (a) The effect is "infinite cost" if their vault is liquidated
            (b) Otherwise they can actually get more utility for maintaining healthy collateral
    (3) Dynamics constraints which dictate net changes / burning and minting
'''
class IpSpeculatorResponse:


    def __init__(self,
            initial_supply,
            initial_bank,
            initial_rate,
            terminal_rate,
            horizon,
            demand_forecast,
            collateral_forecast,
            target_traj,
            penalty_weight
    ):
        ''' Set problem'''
        self.S0 = initial_supply
        self.N0 = initial_bank
        self.a0 = initial_rate
        self.aT = terminal_rate
        self.T  = horizon
        self.Dt = demand_forecast
        self.Ct = collateral_forecast
        self.rvec = []  # These are return ratios based on Ct forecast
        self.delta_ref = target_traj
        self.c = []
        self.step = 0
        self.init_supply = lambda x: x[0]-self.S0
        self.init_bank = lambda x: x[1]-self.N0
        self.init_rate = lambda x: x[2]-self.a0
        self.x0 = []
        self.K = penalty_weight
        self.results = [[],[],[],[],[]]
        self.bounds = [(0,self.S0)]*(5*self.T)
        # Speculator Utility / Ref Deviation Cost
        self.obj_jit = jit(self.objective)
        self.obj_grad = grad(self.obj_jit)  # objective gradient
        self.obj_hess = jacrev(jacfwd(self.obj_jit)) # objective hessian
        
        ## --- Set Problem ------------
        # self.init_bounds()
        self.calculate_returns()
        self.rollout_init_guess()    
        self.set_constraints() 


    def objective(self,x):
        '''Stagewise State Reference :
           --------------------------------------------
        [x1 , x2 , x3 , x4 , x5]  = 
                    [St , Nt, alpha_t , delta , d_alpha] 
        '''
        jnp.asarray(x)
        cost, r = 0 , 1.001
        for i in range(self.T-1):
            beta = (x[(5*i) + 2] + x[(5*i) + 4])
            ## some return idea
            r = self.rvec[i]
            # Trader utility
            cost += r*(x[(5*i)+1] + x[(5*i)+3])\
                        - (x[(5*i)+2] + x[(5*i)+4])*x[5*i]       # Trader utility
            # Penalty on delta deviation
            cost -= self.K*(x[5*i + 3] - self.delta_ref[i])**2  
            # Vault Soft Constraint Penalty, log positve -> closer to 0 = -inf cost
            # cost += jnp.log( 
            #     (1/beta) * (\
            #         ( self.Ct[i]*x[(5*i)+1]  + x[(5*i)+3] ) - (x[5*i] + x[(5*i)+3])
            #     )
            # )
        
        # Terminal Cost
        # cost += np.log(1) + x[(5*self.T) + 2] + x[(5*self.T + 5)] -

        return -1*np.sum(cost)

    def calculate_returns(self):
        for i in range(len(self.Ct)-1):
            self.rvec.append(self.Ct[i+1] / self.Ct[i])
        self.rvec.append(self.rvec[-1])
        return
    
    def init_bounds(self):
        self.bounds[0] = (0, 2*self.S0)
        self.bounds[1] = (0, 2*self.N0)
        self.bounds[2] = (.5, 2)
        self.bounds[3] = (-(1/10)*self.S0, (1/10)*self.S0)
        if self.a0 < self.aT:
            self.bounds[4] = (.01*self.a0, .25*self.a0)
        else:
            self.bounds[4] = (-.25*self.a0, -.01*self.a0)

    def rollout_init_guess(self):
        x =[]
        rate_incr = (self.aT-self.a0)/(self.T-1)
        x[0:2] = [self.S0 , self.N0 , self.a0] 
        x = x+[0]*(self.T-1)*5
        for i in range(self.T-1):
            # moderate guesses
            x[(5*i) + 3] = (self.delta_ref[i] +.001) / 2
            x[(5*i) + 4] = rate_incr
            # satisfy constraints 
            x[(5*i) + 5] = x[(5*i)+3] + x[(5*i)+0]
            x[(5*i) + 6] = x[(5*i)+1] + \
                        (self.Dt[i]*x[(5*i)+3])/(x[(5*i)]*self.Ct[i])
            x[(5*i) + 7] =  x[(5*i)+2] + rate_incr
  
        x[-1] = self.aT  # satisfies alpha terminal constraint
        self.x0 = x
        return

    def supply_constr(self, idx):
        ## map which returns & encodes functions
        i  = int(5*idx) 
        ## Supply Constraint
        return lambda x : x[i+5]-x[i+3]-x[i]
    
    def bank_constr(self, idx):
        ## map which returns & encodes functions
        i  = int(5*idx) 
        D  = self.Dt[idx]
        pC = self.Ct[idx]
        return lambda x: x[i+6] - x[i+1] - (D*x[i+3])/(x[i]*pC)
    
    def rate_constr(self,idx):
        i  = int(5*idx) 
        return lambda x : x[i+7]-x[i+4]-x[i+2]

    def set_constraints(self):
        ## NOTE Initial Conditions ---------------------------------------------
        Hs = lambda x,v : v[0]*jacrev(jacfwd(self.init_supply))(x)
        Hn = lambda x,v : v[0]*jacrev(jacfwd(self.init_bank))(x)
        Ha = lambda x,v : v[0]*jacrev(jacfwd(self.init_rate))(x)
        self.c = [{'type': 'eq', 'fun': self.init_supply, 'jac': jacfwd(self.init_supply), 'hess': Hs}, 
                  {'type': 'eq', 'fun': self.init_bank,   'jac': jacfwd(self.init_bank),   'hess': Hn}, 
                  {'type': 'eq', 'fun': self.init_rate,   'jac': jacfwd(self.init_rate),   'hess': Ha}]
        ## NOTE Dynamics Constraints ---------------------------------------------------------  
        ##   --> lambdas need to apply exact indexing, use map function
        indices   = [ i for i in range(0,(self.T-1))]
        funcs     = list(map(self.supply_constr, indices)) +\
                    list(map(self.bank_constr,   indices)) +\
                    list(map(self.rate_constr,   indices))
        jacobians = [jacfwd(func) for func in funcs]
        hessians  = [jacrev(j) for j in jacobians]
        for i in range(len(funcs)):
            self.c.append({'type': 'eq', 'fun': funcs[i], 'jac': jacobians[i],\
                           'hess': lambda x,v : v[0]*hessians[i](x)})
        ## NOTE: All Terminal Constraints
        ## --- Terminal Rate Constraint ----------------------------------------------
        target_rate = lambda x : (x[-1] - self.aT)
        J = jacfwd(target_rate)
        hess = jacrev(J)
        Hvp = lambda x, v : v[0]*hess(x)
        self.c.append( {'type': 'eq', 'fun': target_rate, 'jac': J, 'hess': Hvp})
        # NOTE: try a different approach where I pass the function
        #           as a single vector func and see if cyipopt can handle this(?)
        ## NOTE Box constraints -- should be dynamic(?) -----------------------
        for i in range(0,self.T):
            count = 1
            if (i == 0):
                self.bounds[(5*i)]   = (0,2*self.S0)
                self.bounds[(5*i)+1] = (0,2*self.N0)
                # self.bounds[(5*i)+2] = (0.5,2*self.a0)
                self.bounds[(5*i)+3] = (-.1*self.S0 , .1*self.S0)
            else:
                count +=1
                self.bounds[(5*i)]   = (0,2*self.bounds[5*(i-1)][1])
                self.bounds[(5*i)+1] = (0,2*self.bounds[5*(i-1)+1][1]) # or some risk constraint
                self.bounds[5*i + 3] = (-.1*self.bounds[5*(i-1)][1],\
                                         .1*self.bounds[5*(i-1)][1]) # relative to current supply
            self.bounds[5*i + 2] = (0.5, 2*self.a0)
            if self.aT > self.a0:
                self.bounds[5*i + 4] = (0,.25*self.a0)
            else:
                self.bounds[5*i + 4] = (-.25*self.a0,0)
        # clip the last 2 boundary vars, not needed
        self.bounds = self.bounds[:-2]

    def solve(self):
        '''wraps minimize, parses & return. catch errors'''
        res = minimize_ipopt(self.obj_jit, jac=self.obj_grad, hess=self.obj_hess, 
                                x0=self.x0,  bounds = self.bounds, constraints=self.c, 
                                options={'disp': 5, 'max_iter':1200, 'tol': 10e-3})
        x = [np.round(i,5) for i in res.x]
        if not res.success:
            print("Reason : ", res.info["status_msg"])
        else:
            print("Success: ", res.success)
        self.sol = x
        return self.parse(x)
    
    def parse(self,sol):
        self.results[0] = sol[0::5]  
        self.results[1] = sol[1::5]
        self.results[2] = sol[2::5]
        self.results[3] = sol[3::5] 
        self.results[4] = sol[4::5]
        return self.results

    def test_constraints(self):
        x = self.sol
        for i in range(self.T-1):
            # workaround for indexing troubles
            s  = int(5*i)
            n  = int((5*i) + 1)
            a  = int((5*i) + 2)
            ds = int((5*i) + 3)
            da = int((5*i) + 4)
            rate_constr = lambda x: x[a+5] - x[a] - x[da]
            supply_constr = lambda x: x[s+5]-x[s]-x[ds]

            print("Rate Constr = 0 ? :" , rate_constr(x))
            print("Supply Constr = 0 ? :" , supply_constr(x))

### NOTE: {Protocol Description}
'''
    This one is a little more straightforward. We are just going to have
    protocol poduce a smooth trajectory to some terminal target, with 
    a quadratic cost function.
'''
class NStepLQPeg:
    ''' The LQ simplification allows us to generate
        an n-step burn/mint trajectory. Furthermore, I put
        arbitary cost on u_net which could be adjusted but this 
        asks the controller to make smooth changes to supply.'''
    def __init__(self, curr_supply, horizon, forecast):
        self.S0 = curr_supply
        self.T = horizon
        self.D_pred = forecast
        self.lamb =  5   #arbitrary for now , control weight
        '''NOTE: Demand forecast is this formulation is added to cost function directly which 
        seemed to work fine, but will imply the same thing: an indirect target on price ''' 
    def solve(self):
        ''' In a way, this is the most trivial MPC formulation we can have'''
        x = cp.Variable((1, self.T + 1))  # n = 1 as state is viewed as supply only
        u = cp.Variable((1, self.T))      # m = 1 as we just solve for delta u
        A , B = 1 , 1
        cost = 0
        constr = []
        for t in range(self.T):
            cost += cp.sum_squares(x[:, t + 1] - self.D_pred[:,t+1])\
                    + self.lamb * cp.sum_squares(u[:, t])
            constr += [x[:, t + 1] == A * x[:, t] + B * u[:, t]]   
            # cp.norm(u[:, t], "inf") <= 1] # possible control constraint
        # Terminal constraint says to match the forecast
        constr += [x[:, self.T] == self.D_pred[:,self.T], x[:, 0] == self.S0]
        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve()
        
        return (np.array(x.value) , np.array(u.value))

def make_some_tests(testnum, K):
    ## NOTE : Most Basic Problem
    if (testnum == 1):
        T = 2
        target = [15]
        problem = IpSpeculatorResponse( initial_supply= 250,
                                    initial_bank  = 70,
                                    initial_rate = 1.5,
                                    terminal_rate = 1.6,
                                    horizon = T,
                                    demand_forecast =   [250] ,
                                    collateral_forecast = [25],
                                    target_traj = [15],
                                    penalty_weight = K
                                )
    if (testnum == 2):
        T = 3
        target = [15, 15]
        problem = IpSpeculatorResponse( initial_supply= 250,
                                    initial_bank  = 70,
                                    initial_rate = 1.5,
                                    terminal_rate = 1.6,
                                    horizon = T,
                                    demand_forecast =   [250,250] ,
                                    collateral_forecast = [25,25],
                                    target_traj = target,
                                    penalty_weight = K
                                )
        
    if (testnum == 3):
        ## NOTE : 5 Time Step Horizon , check simple scaling
        target = [ -8 , -3, -.7 , 0 ]
        T = 5
        problem = IpSpeculatorResponse( initial_supply= 250,
                                        initial_bank  = 70,
                                        initial_rate = 1.5,
                                        terminal_rate = 1.8,
                                        horizon = T,
                                        demand_forecast = [250,250,253,258,257],
                                        collateral_forecast = [20, 21, 16, 18, 19],
                                        target_traj = target,
                                        penalty_weight = K
                                    )

    # #
    # T = 10
    # D = [100,100,99,98,101, 100, 100, 100, 100, 101]
    # Ct = [20, 21, 16, 17, 12, 12, 12,12, 14, 14]
    # target = [ -3 , -3, -3, -1, -1, -1, -1 , 0, 0, 0]
    # penalty = 10
    # problem = IpSpeculatorResponse( initial_supply= 100,
    #                                 initial_bank  = 50,
    #                                 initial_rate = 1.1,
    #                                 horizon = 10,
    #                                 demand_forecast = D,
    #                                 collateral_forecast = Ct,
    #                                 target_traj = target,
    #                                 penalty_weight = penalty
    #                                )

    return problem , target
    
#### NOTE: Probably ARCHIVE these, not particularly useful ############################
## Some Utilites --------------------------------------------------------
def block_diagonal(H , n ):
    stack = []
    for i in range(n):
        A = np.zeros( [3,3*n] )
        A[:,3*i : 3*(i+1)] = H
        stack.append(A)

    A = np.concatenate(stack)
    return A

def copy_gradient(G, T):
    n = len(G)
    stack = [G[i] for i in range(n)]*T
    print(stack)

def shift_supply_constraint(T):
    c = [-1, 0, 0, 1, 0 , -1]
    z = np.zeros(3*T)
    c_init = z
    c_init[0:3] = [1,0,-1]
    stack = [c_init]
    for i in range(T-1):
        print(i)
        c_next =  np.zeros(3*T)
        c_next[3*i:3*(i+2)]= c
        stack.append(c_next)
    return stack

### Optim Problems as individual Classes --------------------------------
class OneStepPeg:

    ''' Simplified optimization problem:
    decision : x = [ Supply, Price, delta Supply] , demand is external
    Cost     : ( x[1] - 1)**2
    subject to  Simple Dynamics:
    x0 = x0 + x2
    x1*x0 = D_t (given demand)
    ----------------------------
    Nonlinear constraint, we need ip method (I think) 
    '''

    def __init__(self, target=1):
        self.target = target
        pass

    def objective(self, x ):
        return  (x[1] - self.target)**2

    def gradient(self, x):
        return np.array([  0,  2*(x[1] - self.target),  0 ])

    def constraints(self, x):
        '''Equality Constraints:
                c1: x0 - u_net = x0_old , 
                c2: x0*x1 = D_t /
            Inequality COnstraints:
                -x0 < u_net  (lower bound implied)
                x0 > 0'''
        return np.array(( x[0] - x[2] , x[0]*x[1] ))

    def jacobian(self, x):
        return np.concatenate(([1, 0, -1], [x[1], x[0], 0]))

    def hessianstructure(self):
        return np.nonzero(np.tril(np.ones((3, 3))))

    def hessian(self, x, lagrange, obj_factor):
        H = obj_factor*np.array((
                (0, 0, 0),
                (0, 2, 0),
                (0, 0, 0) ))
        H += lagrange[0]*np.zeros((3,3))
        H += lagrange[1]*np.array((
                (0, 1, 0),
                (1, 0, 0),
                (0, 0, 0) ))
        row, col = self.hessianstructure()
        return H[row, col]

    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):
        pass
        # print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))

class OneStepPegPenalty:

    ''' Simplified optimization problem , but with control penalty
    decision : x = [ Supply, Price, delta Supply] , demand is external delta = u_net
    Cost     : ( x[1] - 1)**2  + (u_net)**2 
    subject to  Simple Dynamics:
    x0 = x0 + x2
    x1*x0 = D_t (given demand)
    ----------------------------
    Nonlinear constraint, we need ip method (I think) 
    '''

    def __init__(self, target=1):
        self.target = target
        pass

    def objective(self, x ):
        return  (x[1] - self.target)**2  +  (x[2])**2

    def gradient(self, x):
        return np.array([  0 ,  2*(x[1] - self.target),  2*x[2] ])

    def constraints(self, x):
        '''Equality Constraints:
                c1: x0 - u_net = x0_old , 
                c2: x0*x1 = D_t /
            Inequality COnstraints:
                -x0 < u_net  (lower bound implied)
                x0 > 0'''
        return np.array(( x[0] - x[2] , x[0]*x[1] ))

    def jacobian(self, x):
        return np.concatenate(([1, 0, -1], [x[1], x[0], 0]))

    def hessianstructure(self):
        return np.nonzero(np.tril(np.ones((3, 3))))

    def hessian(self, x, lagrange, obj_factor):
        H = obj_factor*np.array((
                (0, 0, 0),
                (0, 2, 0),
                (0, 0, 2) ))
        H += lagrange[0]*np.zeros((3,3))
        H += lagrange[1]*np.array((
                (0, 1, 0),
                (1, 0, 0),
                (0, 0, 0) ))
        row, col = self.hessianstructure()
        return H[row, col]

    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):
        pass
        # print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))
############################################################################
    


if __name__ == "__main__":
    res = []
    penalty = 2
    # NOTE : setup and check test problems 
    prob, target, = make_some_tests(TEST, K = penalty)
    print("Assert ( Num Vars = {} = Bounds ) : ".format(len(prob.x0))\
                    , (len(prob.x0) == len(prob.bounds)) )
    print("Solution Guess:  " , prob.x0, "\n")
    print("Init Bounds: " , prob.bounds)
    print("Num Constraints: ", len(prob.c))


    if (SOLVE):
        res = prob.solve()
        prob.test_constraints()
        print("Supply Traj : ", res[0])
        print("Bank Traj : "  , res[1])
        print("Fee Traj :: ", res[2])
        print("Burn/Mint Traj.", res[3])
        print("Rate Traj. ", res[4])



    ## --> NOTE : Plotting Problem Responses  ---------------------------------
    if (PLOT):
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(2,2)
        ax[0,0].plot(res[0]) 
        ax[0,1].plot(res[1])
        ax[1,0].plot(res[2])
        ax[1,1].plot(res[3], color = "tab:blue", linewidth =2)
        ax[1,1].plot(target[:-1], color ="firebrick", linewidth =2)
        ax[1,1].legend(['spec choices', 'protocol target'], loc = 'best')
        ax[0,0].set_title("Supply Change")
        ax[0,1].set_title("Bank Change")
        ax[1,0].set_title("Rate Change")
        ax[1,1].set_title("Burn /Mints")
        fig.suptitle("Speculator Choices, Penalty Factor: {}".format(penalty))
        fig.tight_layout()

        plt.show()
