package lab.demo_ai;

import org.apache.ibatis.annotations.Mapper;

import java.util.List;
import java.util.Map;

/**
 * [기능요약] MyBatis Mapper 호출을 통한 DB 연동 Repository
 *
 * [상세설명]
 * - MyBatis XML 매퍼를 통해 DB의 `member` 테이블을 조작
 * - 계정 정보 조회, 단건 조회, 등록, 수정, 비밀번호 초기화, ID 중복 체크 기능을 제공
 */
@Mapper
public interface AccountRepository {
    List<Map<String,Object>> sltAccountInfo(Map<String, Object> param);
    Map<String,Object> sltAccountOne(Map<String, Object> param);
    void istAccount(Map<String, Object> param);
    void uptAccount(Map<String, Object> param);
    void resetPw(Map<String, Object>param);
    Map<String, Object> dupCheck(Map<String, Object> param);
}
