### FH03 Senior Design Project

## Components

### UT Coin Component (Protocol)
- Features our own desing of protocol actions
- Input: This should accept a price (demand) forecast and current state data : total supply, current price, collateralization level
- Output: Time series of actions taken by our protocol which is also propagated out by speculator responses
- ( DAI-"ish" model)

#### Testing: Monte Carlo Simulations
####### Feedback/ Let me Know : Does this make sense how we want to use it ?
- Why :
  - What-if analysis if we propagate out different time series across time
  - stress test, how many cases do we have failures ? Quantify distance from our peg, how sever are price changes ?
  - Open Question : developing metrics of performance ( rate of price change, volume , etc. )
- Result : How does our controller model act in response to what-if, quantify how often we are stable / not

- Input : N time series of ETH  
- Output : N "expected" paths of supply change -- which simulates our protocol (deterministic) response
- Sources of Uncertainty
-- We have uncertainty in future ETH prices ( that we can hopeflly capture in distribution of MC simulation )
-- We also have uncertainty in "market response" -- this is how our agent responds to the changes that the protocol makes

### Real Coin Forecasting Component (?)
###### Question : 
- This component, unlike UT Coin, is not necessarily trying to do anything different than exisiting stablecoins but rather attempt an accurate (simplified) model of current coins ... either by using time series prediction only or some simple utility based models
- Why:
   - Stress test for market predictions. This would be more of a pure statistical analysis approach
   - So given a "what-if" (fake) set of starting inputs such as:
   -   Current Supply
   -   LP levels (maybe)
   -   Collateralization Level
   -   Estimation of Number of Users
   -   Size of Certain Participants
   -   Time Series Predictions/ Expectations ( Correlated Assets such as BTC, ETH ) (idea of future demand for MC sims)
  - Output:
  - Again an MC analysis of possible outcomes, where we can try to quantify sensitivity of real coins to certain events
 - How: As much as possible want to try to reuse a couple basic models to have some coverage
 -   DAI:
 -     Steal DAI model from paper you guys found -- we vary those parameters (stability fee and collateralization ratio)
 -     Closest to achievable
 -   USDC:
 -     Need to work more with MITRE on this. Model may be completely different ( Sanith hinted size of cash flows)
 -     Mitre interested in this
 -     May also have a lot to do with riskiness of partners
 -   Terra/Luna
 -     Might have some similarities with DAI. This one is honestly quite tricky because it depends a lot on predicting how people will "feel" about future prices. It seems to be built a lot on expectation of success
 -     Simplest thing to do might just be take past prices ( of seigniorage + stablecoin ) and output new stablecoin predictions
 -   Fei
 -     This is a model that may also be covered by our speculator model, but we will add to it the existence of liquidty pools
 -   Rai 
