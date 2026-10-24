class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        # x_curr = init
        
        # for iter in range(iterations):
        #     f_prime = 2 * x_curr
        #     x_curr = x_curr - learning_rate*f_prime
            
        # return round(x_curr, 5)
        if iterations == 0:
            return init
        return round(((1 - 2*learning_rate)**iterations)*init, 5)