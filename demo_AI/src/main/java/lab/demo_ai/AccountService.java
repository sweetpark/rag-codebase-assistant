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

    /**
     * [기능요약] 계정 관련 서비스 로직 실행
     *
     * [상세설명]
     * - `oper`값에 따라 계정 조회, 등록, 수정, 중복확인, 비밀번호 초기화 로직 분기 실행
     * - Repository 계층을 호출하여 실제 DB 작업을 수행
     *
     * @param oper 작업 유형 (read, readOne, regist, edit, dupCheck, reset)
     * @param payload 클라이언트로부터 받은 데이터
     * @return 실행 결과 map
     */
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
