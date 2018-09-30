--[[
    The setup class aggregates a number of tasks in a single experiment.
    It contains methods implementing the "outermost" loops for training.
    I use it also for training with a single task.
    Members:
    -- tasks - a table of Task objects
--]]
local Setup = torch.class('NetworkTraining.Setup')

function Setup:__init(tasks)
  self.total_iter=0  -- counts "outermost" loops
                     -- when using "Train" this is equivalent to # updates
  self.tasks=tasks
  -- check that all the tasks conform to the "TaskInterface"
  for k,v in pairs(self.tasks) do 
    NetworkTraining.check_object_interface_compatibility(NetworkTraining.TaskInterface,v)
  end
end

function Setup:Train(iters)
--[[
  the main function used for training a network in a multi-task scenario
  first, the gradient with respect to weights is computed for all tasks
  then, the weights are updated for all the tasks
  this is the method to use when networks in two tasks share weights
  arguments:
    -- iters - the number of iterations (forward-backward-update) to perform
--]]
  local t=torch.tic()
  self.local_iter=0
  while self.local_iter<iters do
    -- local tt=torch.tic()
    for i_task=1,#self.tasks do
      self.tasks[i_task]:perform()
    end
    -- print(" setup:train perform time",torch.toc(tt))
    for i_task=1,#self.tasks do
      self.tasks[i_task]:update()
    end
    self.local_iter=self.local_iter+1
    self.total_iter=self.total_iter+1
    if torch.toc(t)>3 then -- dont update the progress bar too often
      xlua.progress(self.total_iter,self.total_iter+iters-self.local_iter)
      t=torch.tic()
    end
    -- print("total setup:train loop time",torch.toc(tt))
  end
end

function Setup:TrainSequential(iters,multp,clip)
--[[
  in this function the updates on the shared weights are performed
  in a sequential manner. A gradient is calculated for one task, 
  then the weight for that task is updated,
  the process is repeated for the next task.
  arguments
    -- iters - number of iterations to perform
    -- multp - a table where key corresponds to task index 
               and value is number denoting the requested number of updates
               for the corresponding task in a single outermost iteration
               its useful for example for Wasserstein GAN, 
               where many discrimination updates are needed 
               for a generator update
    -- clip  - a table where key corresponds to task index and value is a number
               to which the absolute value of each weight is clipped.
               ( clipping is max(-c,min(c,x)) where c is the 
               clipping value, and x is the value being clipped )
-]]
  local t=torch.tic()
  self.local_iter=0
  while self.local_iter<iters do
    for i_task=1,#self.tasks do
      local niter=1
      if multp and multp[i_task] then niter=multp[i_task] end
      for i_iter=1,niter do
        -- forward-backward
        self.tasks[i_task]:perform()
        -- weight update
        self.tasks[i_task]:update()
        if clip and clip[i_task] then 
          self.tasks[i_task]:clipParams(clip[i_task]) 
        end
      end
    end
    self.local_iter=self.local_iter+1
    self.total_iter=self.total_iter+1
    if torch.toc(t)>3 then -- dont update the progress bar too often
      xlua.progress(self.total_iter,self.total_iter+iters-self.local_iter)
      t=torch.tic()
    end
  end
end
