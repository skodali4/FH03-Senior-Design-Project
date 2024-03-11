""" {Description}

This file should have a lot of similarities with the DAI file however we are probably actually
not going to use the deleveraging library to simulate or run a RAI test. Instead, we should most likely
use a variant of the UTCoin code. However leave that part for now. 

The main thing to work on would be setting up the PID controller.

Specifically read the section of RAI's whitepaper on "control theory" --> (Link)

Note how they are using the PID controller. We want to emulate that behavior with the same underlying
basic dynamics model for DAI stability wiht this small addition on top. 

Some python library options for this would be simple-pid (https://pypi.org/project/simple-pid/) or the control toolbox. This just saves
use the time and effort of building one from scratch and we can instead focus on tuning for our simulations. I would say probably we should tune
the PID to try to match historical times series or maybe we should go from first principles and just try to do it the way it has been described in
their whitepaper.

To be more explicit, note the "setpoint" parameter in the example below from simple-pid's pypi site. 

from simple_pid import PID
pid = PID(1, 0.1, 0.05, setpoint=1)

# Assume we have a system we want to control in controlled_system
v = controlled_system.update(0)

while True:
    # Compute new output from the PID according to the systems current value
    control = pid(v)
    
    # Feed the PID output to the system and get its current value
    v = controlled_system.update(control)

Setpoint should esseentially match whatever it 
is that Rai's system is targetting. They say this in their paper, but off memory the target has to do with an ideal rate of change in the price 
rather than the price itself. This gives us a way of tuning ( because we can use RAI data directly ). The input is RAI (index) price. The parameters 
P, I and D need to be tuned. Need to check the paper for details but this should back out into a stability fee calculation that we should use to simulate
the system forward a few time steps ( exact details need to be worked out )

"""

## TODO : Implement PID on top of DAI code for a RAI model

def get_inputs()
  """ should work exactly the same way as the other function in dai template file
        and should interface with the UI team's framework"""
  data = uicall()
  return data

def simulate()
  
  data = get_inputs()
  T = data.horizon #(user inputs)
  P_t = data.price #(from demand etc.)
  alpha = data.init_rate
  
  # iteratvie response 
  agent = RaiSpeculator() 
  protocol = ReflexerLabs()
  price_dist = []
  
  for n in data.num_simulations():
    price_series = []
    
    for t in range(T):
      alpha = protocol.set_rate(P_t)
      agent.solve(alpha)
      # update market paremeters
      # Supply
      # Demand
      price_series.append(P_t)
      # extract new price
      P_t = None #(new)
      
    price_dist.append(price_series)
  
  return



class RaiSpecualtor:
  def __init__(self):
    # TODO
    return
  
  def solve():
    #TODO
    # most likely integrate with UT coin problem
    # utcoin_speculator( alpha = new_data )
    return


class RefelxerLabs:
  def __init__(self):
    #TODO
    #self.contoller = init_pid 
    return
  
  def tune_pid():
    return

  def update_setpoint():
    # new setpoint
    #self.controller.setpoint = new
    return
  

