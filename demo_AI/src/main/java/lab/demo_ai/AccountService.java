package lab.demo_ai;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Transactional
@Service("account")
public class AccountService implements baseTx {

    @Autowired
    AccountRepository accountRepository;

    @Override
    public Map<String, Object> execute(Map<String, Object> inParam) {

        Map<String, Object> resultMap = new HashMap<>();

        try{
            if(inParam.get("oper") != null){
                if(inParam.get("oper").equals("read")){
                    List<Map<String, Object>> dataMap = accountRepository.sltAccountInfo(inParam);

                    resultMap.put("data", dataMap);
                    resultMap.put("res_cd", "0000");
                    resultMap.put("res_msg", "success");


                }else if(inParam.get("oper").equals("readOne")){
                    Map<String, Object> dataMap = new HashMap<>();
                    dataMap = accountRepository.sltAccountOne(inParam);

                    resultMap.put("data", dataMap);
                    resultMap.put("res_cd", "0000");
                    resultMap.put("res_msg", "success");
                } else if(inParam.get("oper").equals("regist")){
                    accountRepository.istAccount(inParam);

                    resultMap.put("res_cd", "0000");
                    resultMap.put("res_msg", "success");

                }else if(inParam.get("oper").equals("edit")){

                    accountRepository.uptAccount(inParam);
                    resultMap.put("res_cd", "0000");
                    resultMap.put("res_msg", "success");

                }else if(inParam.get("oper").equals("reset")){

                    accountRepository.resetPw(inParam);
                    resultMap.put("res_cd", "0000");
                    resultMap.put("res_msg", "success");

                }else if(inParam.get("oper").equals("dupCheck")){
                    resultMap.put("data", accountRepository.dupCheck(inParam));
                    resultMap.put("res_cd", "0000");

                }
            }
        }catch(Exception e){
            resultMap.put("res_cd" , "9999");
            resultMap.put("res_msg", "오류가 발생했습니다");
            throw e;
        }



        return resultMap;
    }
}
