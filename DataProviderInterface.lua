
--[[
  the interface between the "Task" and the "DataProvider" class.
  the task represents a single learning task/objective,
  with its own data, objective function and specific logging mechanism

  the "logger" class is responsible for logging training performance,
  test results and information concerning the updates

  the "Task" does not depend on any functionality exposed by the logger;
  rather, it exposes to the logger the information that might be used
  to create the contents of log files
--]]

NetworkTraining.DataProviderInterface={
  getInput={
    desc="returns the input data, previously loaded by \"getBatch\";\n "
  },
  getTarget={
    desc="returns the ground truth predictions, "..
         "previously loaded by \"getBatch\";\n "
  },
  getBatch={
    desc="prepare a new batch of input data and ground truth labels "..
         "these can then be exposed by \"getInput\" and \"getTarget\"."..
         "The method should have a mechanism for counting data "..
         "and return false after completion of a full iteration "..
         "over the test set. In this case the data does not "..
         "need to be loaded. Else it should return true.\n"..
         "argument:\n "..
         "  number of data items expected in the batch\n"..
            "note, that the last testing batch can be smaller than requested\n"
  }
}
