--[[
  the interface between the "setup" and the "task" class.
  the task represents a single learning task/objective,
  with its own data, objective function and specific logging mechanism

  the "setup" class organizes training with a single or multiple tasks

  it calls the perform() and update() methods on each "task" object
  the "perform" normally computes a gradient w.r. to weights
  and the "update" updates the weights w.r. to the gradient
  These two operations are separated intentionally, because when two tasks
  are sharing some weights, you want to compute the gradient for both tasks first
  and only then update the weights.

  if clipping is requested in the "TrainSequential" method of class "setup"
  the implementation of "Task" needs to expose a "clipParameters" method
  this is not used if clipping is not requested in a call to "TrainSequential"
  ( clipping is max(-c,min(c,x)) where c is the 
    clipping value, and x is the value being clipped )
--]]

NetworkTraining.TaskInterface={
  perform={
    desc="The function that computes a gradient of the task objective funciton"..
         " with respect to predictor parameters"
  },
  update={
    desc="The function that updates predictor weights using the gradient "..
         " previously computed by the \"perform\" method "
  }
}
