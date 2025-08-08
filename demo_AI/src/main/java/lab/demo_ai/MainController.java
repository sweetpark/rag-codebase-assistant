package lab.demo_ai;

import lombok.extern.slf4j.Slf4j;


import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;

import java.util.HashMap;
import java.util.Map;


/**
 * [기능요약] 계정 관련 API 요청을 처리하는 컨트롤러
 *
 * [상세설명]
 * - 프론트에서 전달된 서비스명(`account`)과 작업 유형(`oper`)을 기반으로 요청을 분기 처리
 * - `AccountService`를 통해 실제 로직을 실행하며 결과를 응답으로 반환
 *
 * @param requestBody 요청 파라미터 (service, oper, payload)
 * @return JSON 형태의 결과 응답
 */

@Controller
@Slf4j
public class MainController {

    @Autowired
    ApplicationContext ac;

    @PostMapping("/api/main")
    public ResponseEntity<Map<String, Object>> mainController(@RequestBody Map<String, Object> inParam){
        Map<String,Object> resultMap = new HashMap<>();

        log.info("inParam : " + inParam);
        baseTx account = (baseTx) ac.getBean(inParam.get("service").toString());
        Map<String, Object> param = (Map<String, Object>) inParam.get("payload");
        resultMap = account.execute(param);


        return ResponseEntity.ok(resultMap);
    }
}
