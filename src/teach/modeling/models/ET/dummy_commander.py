class DummyCommander(object):
    def __init__(self):
        super().__init__()
        self.turns = 0

    def reset(self):
        self.turns = 0
        return 

    def extract_progress_check_subtask_string(self, pc_result):
        if pc_result["success"]:
            return ""
        
        for subgoal in pc_result["subgoals"]:
            if subgoal["success"] == 1:
                continue 
            
            if subgoal["steps"][0]["success"] == 0 and len(subgoal["description"])>0:
                return subgoal["description"]
            
            for step in subgoal["steps"]:
                if step["success"] == 1: 
                    continue 
                
                if len(step["desc"])>0:
                    return step["desc"]
    
    def step(self, input_dict, vocab, prev_action, agent="commander", **kwargs):
        obj_cls = 0
        self.turns += 1 

        if len(input_dict['driver_action_history']) > 0:        
            driver_last_action = input_dict['driver_action_history'][-1]
        
        if self.turns == 0:
            action_output = dict(
                action="Text",
                obj_cls=obj_cls,
                utterance="Hello!"
            )

            return action_output
        elif self.turns == 1:
            print('Forcing progress check!')
            action = 'OpenProgressCheck'
        elif (
            prev_action == "OpenProgressCheck"
            or not driver_last_action['success']
            or driver_last_action['action'] == "Text"
        ):
            # If driver failed last action, provide it next subgoal
            action = "Text"
        else:
            action = "NoOp"

        # Simple commander speaker that uses the text from progress check output
        utterance = None
        if action == "Text":
                
            # Get pc results
            pc_results = input_dict['commander_action_result_history'][-1]

            task_desc = kwargs['tatc_instance']['tasks'][0]['desc']
            next_instr = self.extract_progress_check_subtask_string(pc_results)

            # Check if goal already provided 
            goal_provided = False 
            for sent, _ in input_dict['dialogue_history']:
                if task_desc == sent:
                    goal_provided = True 

            instr = task_desc if not goal_provided else next_instr
            utterance = instr
        
        action_output = dict(
            action=action,
            obj_cls=obj_cls,
            utterance=utterance
        )
        return action_output