class BIO:
    def __init__(self):
        pass

    def verify_context(self,context):
        context = context.replace("."," .")
        context = context.replace(","," ,")
        context = context.replace("?"," ?")
        return context

    def find_indexes(self,context_list,answers_list):
        
        start_index = -1
        end_index = -1
        
        indexes_found = False
        
        i=0
        # print(context_list," \n",answers_list)
        while i < len(context_list):
    
            for ans_index,ans_word in enumerate(answers_list):
                temp=context_list[i]
                if ans_index == 0:
                    
                    if context_list[i] ==  ans_word:
                        # print(context_list[i])
                        start_index =  i 
                        i+=1
                    else:
                        i+=1
                        break
                else:

                    if ans_index==len(answers_list)-1:
                        if context_list[i] ==  ans_word:
                            end_index=i
                            indexes_found=True
            
                    elif context_list[i] !=  ans_word:
                        # print(context_list[i])
                        indexes_found = False
                        
                        break
                    i+=1
                
            if indexes_found == True:
                break         

        return start_index,end_index
    
    
    def convert_BIO(self,context,answers):
        context = self.verify_context(context)
        context_list = context.split()
        answers_list = answers.split()

        start_index,end_index = self.find_indexes(context_list,answers_list)
        bio_sen = ""
        index=0
        while index < len(context_list):
            if index==start_index:
                bio_sen+=context_list[index]+" "+"bans "
                index+=1
                while index < end_index+1:
                    bio_sen+=context_list[index]+" "+"ians "

                    index+=1
            else:
                # bio_sen+=context_list[index]+" "+"oans "
                bio_sen+=context_list[index]+" "
                index+=1
        return bio_sen