# OCR Detector for Floor Plan Text Recognition
# Detects space names and labels in multiple languages
# Enhanced with deduplication and multi-language support

import cv2
import numpy as np
from PIL import Image
import easyocr
import re
from typing import List, Dict, Tuple, Optional
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Comprehensive watermark and non-space keywords to filter out (English and Korean)
WATERMARK_KEYWORDS = [
    # English watermarks and metadata
    'alamy', 'shutterstock', 'getty', 'watermark', 'stock', 'copyright', '©', 'www', 'http', '.com',
    'sample', 'preview', 'demo', 'example', 'template', 'draft', 'concept',
    'architect', 'designer', 'company', 'inc', 'ltd', 'corp', 'llc',
    'property', 'real estate', 'realty', 'development', 'construction',
    'not to scale', 'scale', 'dimensions', 'measurements', 'drawing',
    'blueprint', 'floor plan', 'site plan', 'elevation', 'section',
    
    # Korean watermarks and metadata
    '워터마크', '저작권', '샘플', '견본', '미리보기', '데모', '예시', '템플릿',
    '초안', '컨셉', '건축', '설계', '회사', '부동산', '개발', '건설',
    '축척', '치수', '측정', '도면', '설계도', '평면도', '배치도'
]

# Comprehensive room/space vocabulary for both residential and office environments
class SpaceVocabulary:
    """Comprehensive vocabulary for space names in multiple languages and contexts"""
    
    # English residential spaces
    ENGLISH_RESIDENTIAL = [
        # Basic rooms
        'bedroom', 'living room', 'kitchen', 'bathroom', 'dining room', 'family room',
        'guest room', 'master bedroom', 'master bath', 'powder room', 'half bath',
        'full bath', 'guest bath', 'en suite', 'ensuite',
        
        # Specialized residential spaces
        'foyer', 'entryway', 'mudroom', 'laundry room', 'utility room', 'pantry',
        'walk-in closet', 'closet', 'linen closet', 'coat closet', 'storage',
        'basement', 'attic', 'garage', 'den', 'study', 'library', 'home office',
        'playroom', 'nursery', 'sunroom', 'porch', 'patio', 'deck', 'balcony',
        'terrace', 'conservatory', 'solarium', 'great room', 'bonus room',
        'media room', 'theater room', 'wine cellar', 'workshop', 'craft room',
        'exercise room', 'gym', 'sauna', 'steam room', 'hot tub', 'pool house',
        
        # Fractional bathrooms
        '1/2 bath', '3/4 bath', 'quarter bath', 'half bath', 'powder bath',
        
        # Room variations
        'bed', 'bath', 'br', 'ba', 'rm', 'kit', 'lr', 'dr', 'fr'
    ]
    
    # English office/commercial spaces
    ENGLISH_OFFICE = [
        # Executive and management
        'office', 'executive office', 'ceo office', 'president office', 'director office',
        'manager office', 'supervisor office', 'private office', 'corner office',
        
        # Meeting and collaboration spaces
        'conference room', 'meeting room', 'boardroom', 'training room', 'seminar room',
        'presentation room', 'war room', 'huddle room', 'phone booth', 'collaboration space',
        'team room', 'project room', 'focus room', 'quiet room', 'think tank',
        
        # Work areas
        'open office', 'cubicle', 'workstation', 'desk area', 'bullpen', 'workspace',
        'coworking space', 'hot desk', 'flex space', 'activity based working',
        
        # Reception and lobby
        'reception', 'lobby', 'waiting area', 'entrance', 'foyer', 'atrium',
        'welcome area', 'front desk', 'concierge', 'security desk',
        
        # Support spaces
        'break room', 'kitchen', 'kitchenette', 'cafeteria', 'dining area',
        'coffee bar', 'pantry', 'lounge', 'rest area', 'wellness room',
        'lactation room', 'prayer room', 'meditation room', 'quiet zone',
        
        # Storage and utility
        'storage room', 'supply room', 'filing room', 'archive', 'records room',
        'mail room', 'copy room', 'print room', 'server room', 'it room',
        'electrical room', 'mechanical room', 'janitor closet', 'cleaning closet',
        'utility closet', 'coat room', 'locker room',
        
        # Specialized office spaces
        'laboratory', 'lab', 'research room', 'testing room', 'quality control',
        'design studio', 'creative space', 'workshop', 'maker space',
        'video conference room', 'recording studio', 'broadcast room',
        'control room', 'monitoring room', 'security office',
        
        # Health and safety
        'first aid room', 'medical room', 'nurse station', 'wellness center',
        'fitness room', 'gym', 'shower room', 'changing room',
        
        # Facilities
        'restroom', 'bathroom', 'washroom', 'wc', 'toilet', 'mens room',
        'womens room', 'unisex bathroom', 'accessible bathroom',
        
        # Abbreviations and short forms
        'conf rm', 'mtg rm', 'br rm', 'stor', 'mech', 'elec', 'it',
        'hr', 'admin', 'acct', 'fin', 'ops', 'dev', 'qa', 'rd'
    ]
    
    # Korean residential spaces - Comprehensive list
    KOREAN_RESIDENTIAL = [
        # Basic rooms and variations
        '방', '침실', '안방', '작은방', '큰방', '주침실', '부침실', '자녀방', '아이방',
        '신혼방', '부부방', '게스트룸', '손님방', '객실', '별실', '다다미방',
        
        # Living spaces
        '거실', '응접실', '응접간', '리빙룸', '패밀리룸', '가족실', '홀', '대청마루',
        '마루', '온돌방', '큰방', '작은방', '건넌방', '안채', '사랑채',
        
        # Kitchen and dining
        '부엌', '주방', '조리실', '주방실', '간이주방', '미니주방', '키친', '키친룸',
        '식당', '다이닝룸', '다이닝', '식사공간', '아침식사공간', '조식공간',
        '찬방', '찬간', '식품보관실', '팬트리', '식료품실',
        
        # Bathrooms and hygiene
        '화장실', '욕실', '변소', '세면실', '샤워실', '목욕탕', '목욕실',
        '반욕실', '파우더룸', '화장대', '세면대', '양변기', '비데',
        '사우나', '찜질방', '스팀룸', '온천욕실', '자쿠지룸',
        
        # Entrance and circulation
        '현관', '입구', '현관문', '현관홀', '현관실', '신발장', '신발보관실',
        '복도', '계단', '층계', '계단실', '엘리베이터', '승강기', '홀웨이',
        '통로', '연결통로', '내부통로', '외부통로',
        
        # Storage and utility
        '창고', '저장실', '수납공간', '수납실', '보관실', '물품보관실',
        '다용도실', '세탁실', '빨래방', '세탁공간', '건조실', '다림질실',
        '보일러실', '기계실', '전기실', '설비실', '유틸리티룸',
        '청소용품실', '청소도구실', '걸레보관실',
        
        # Closets and dressing
        '옷장', '붙박이장', '드레스룸', '드레싱룸', '탈의실', '옷방',
        '리넨룸', '이불장', '침구보관실', '계절용품실', '겨울용품실',
        '여름용품실', '의류보관실', '코트룸',
        
        # Outdoor and extension spaces
        '베란다', '발코니', '테라스', '마당', '정원', '앞마당', '뒷마당',
        '옥상', '루프탑', '옥상정원', '테라스정원', '썬룸', '선룸',
        '온실', '유리방', '햇빛방', '일광욕실',
        
        # Garage and parking
        '차고', '주차장', '지하주차장', '실내주차장', '자동차보관소',
        '오토바이보관소', '자전거보관소', '창고겸차고',
        
        # Study and work spaces
        '서재', '공부방', '독서실', '도서실', '서실', '문고',
        '사무실', '홈오피스', '재택근무실', '컴퓨터실', 'PC방',
        '작업실', '작업공간', '아틀리에', '스튜디오', '창작실',
        
        # Entertainment and hobbies
        '취미실', '놀이방', '게임룸', '오락실', '당구장', '탁구장',
        '피아노실', '음악실', '연습실', '악기실', '노래방',
        '영화감상실', '홈시어터', '미디어룸', 'TV룸',
        
        # Exercise and wellness
        '운동실', '헬스장', '홈짐', '요가실', '필라테스실', '트레이닝룸',
        '운동기구실', '러닝머신실', '웨이트룸', '스트레칭룸',
        
        # Special purpose rooms
        '다락방', '다락', '지하실', '지하방', '반지하', '지하공간',
        '펜트하우스', '루프탑룸', '전망대', '조망실',
        '안전실', '대피실', '비상실', '방공호', '지하벙커',
        
        # Traditional Korean spaces
        '한옥방', '온돌방', '마루방', '대청', '누마루', '툇마루',
        '사랑방', '안방', '건넌방', '골방', '부엌간', '찬간',
        '행랑방', '사랑채', '안채', '별채', '문간방',
        
        # Modern apartment terms
        '아파트', '빌라', '원룸', '투룸', '쓰리룸', '포룸',
        '복층', '메이저룸', '알파룸', '베타룸', '펜트룸',
        
        # Room descriptors
        '큰방', '작은방', '넓은방', '좁은방', '밝은방', '어두운방',
        '따뜻한방', '시원한방', '조용한방', '시끄러운방', '전망좋은방',
        '남향방', '북향방', '동향방', '서향방', '모서리방', '가운데방'
    ]
    
    # Korean office/commercial spaces - Comprehensive list
    KOREAN_OFFICE = [
        # Executive offices and management
        '사무실', '개인사무실', '전용사무실', '독립사무실', '개별사무실',
        '임원실', '대표실', '사장실', '회장실', '부회장실', '부사장실',
        '전무실', '상무실', '이사실', '감사실', '고문실', '자문실',
        '부장실', '차장실', '과장실', '팀장실', '실장실', '본부장실',
        '센터장실', '지점장실', '영업소장실', '지사장실', '관리소장실',
        '코너오피스', '모서리사무실', '창가사무실', '내부사무실',
        
        # Meeting and conference spaces
        '회의실', '대회의실', '소회의실', '중회의실', '원형회의실',
        '이사회실', '임원회의실', '주주총회실', '이사회의실',
        '세미나실', '세미나룸', '교육실', '강의실', '연수실', '훈련실',
        '프레젠테이션룸', '발표실', '시연실', '데모룸', '전시실',
        '화상회의실', '원격회의실', '웹회의실', '온라인회의실',
        '미팅룸', '미팅스페이스', '토론실', '협의실', '상담실',
        '브레인스토밍룸', '아이디어룸', '창의실', '기획회의실',
        '고객상담실', '클라이언트룸', '접견실', '면담실',
        
        # Work areas and spaces
        '사무공간', '업무공간', '워크스페이스', '오피스공간',
        '개방형사무실', '오픈오피스', '열린사무실', '공용사무실',
        '칸막이사무실', '파티션사무실', '구획사무실', '분할사무실',
        '큐비클', '워크스테이션', '개인워크스테이션', '팀워크스테이션',
        '데스크', '업무데스크', '개인데스크', '공용데스크', '핫데스크',
        '좌석', '업무좌석', '지정좌석', '자유좌석', '순환좌석',
        '팀공간', '팀룸', '프로젝트룸', '태스크포스룸', '전담팀실',
        '협업공간', '콜라보레이션룸', '공동작업실', '그룹스터디룸',
        '집중업무실', '조용한방', '사일런트룸', '개인집중실',
        
        # Reception and entrance areas
        '접수처', '리셉션', '접수데스크', '안내데스크', '프론트데스크',
        '로비', '현관', '입구', '메인로비', '1층로비', '엘리베이터홀',
        '대기실', '웨이팅룸', '고객대기실', '방문자대기실',
        '안내실', '인포메이션', '종합안내소', '고객센터',
        '보안데스크', '경비실', '수위실', '관리사무소', '시설관리실',
        '출입통제실', '방문자등록소', '게스트체크인',
        
        # Break and dining areas
        '휴게실', '휴식실', '라운지', '임직원라운지', '직원휴게실',
        '카페테리아', '구내식당', '직원식당', '사원식당', '급식실',
        '식당', '식사공간', '다이닝룸', '다이닝스페이스',
        '커피바', '카페', '스낵바', '음료대', '자판기코너',
        '다과실', '티룸', '차실', '커피룸', '음료준비실',
        '주방', '간이주방', '키친', '키친룸', '조리실',
        '팬트리', '식품보관실', '냉장고실', '식료품실',
        '흡연실', '흡연구역', '금연구역', '야외흡연실',
        
        # Storage and utility spaces
        '창고', '보관실', '저장실', '물품보관실', '용품보관실',
        '자료실', '서류보관실', '문서보관실', '아카이브', '기록보관소',
        '파일링룸', '파일보관실', '서류정리실', '문서관리실',
        '우편실', '우편물분류실', '택배보관실', '물류실',
        '복사실', '인쇄실', '출력실', '프린터룸', '복합기실',
        '팩스실', '스캔실', '문서작업실', '제본실', '라미네이팅실',
        '서버실', '전산실', '컴퓨터실', 'IT실', '네트워크실',
        '통신실', '전화교환실', '통신장비실', '라우터실',
        '전기실', '전력실', '배전실', '분전반실', '전기설비실',
        '기계실', '설비실', '공조실', '냉난방실', '보일러실',
        '청소용품실', '청소도구실', '관리용품실', '소모품실',
        '라커룸', '사물함실', '개인사물함', '직원사물함',
        '탈의실', '옷장', '코트룸', '우산보관소',
        
        # Specialized work spaces
        '연구실', '연구개발실', 'R&D실', '기술연구소', '연구센터',
        '실험실', '테스트실', '검사실', '시험실', '분석실',
        '개발실', '소프트웨어개발실', '하드웨어개발실', '제품개발실',
        '디자인실', '디자인스튜디오', '창작실', '아트룸',
        '스튜디오', '작업실', '제작실', '생산실', '공작실',
        '품질관리실', '품질보증실', 'QA실', 'QC실', '검품실',
        '방송실', '녹음실', '녹화실', '편집실', '후반작업실',
        '제어실', '컨트롤룸', '모니터링실', '관제실', '감시실',
        '보안실', '경비실', '감시카메라실', 'CCTV실',
        
        # Training and education
        '교육실', '연수실', '훈련실', '실습실', '체험실',
        '강의실', '강당', '대강당', '소강당', '세미나홀',
        '컨퍼런스홀', '이벤트홀', '다목적홀', '행사장',
        '도서실', '자료실', '정보실', '학습실', '독서실',
        
        # Health and wellness
        '의무실', '보건실', '응급실', '구급실', '간호실',
        '상담실', '심리상담실', '카운슬링룸', '멘탈케어룸',
        '휴식실', '수면실', '낮잠실', '안식실', '명상실',
        '수유실', '모유수유실', '육아휴게실', '맘스룸',
        '기도실', '종교실', '예배실', '명상실', '묵상실',
        '헬스장', '운동실', '피트니스룸', '체육관', '짐',
        '요가실', '필라테스실', '스트레칭룸', '운동기구실',
        '샤워실', '세면실', '탈의실', '사우나', '찜질방',
        
        # Bathrooms and facilities
        '화장실', '남자화장실', '여자화장실', '남녀공용화장실',
        '장애인화장실', '휠체어화장실', '다목적화장실',
        '공용화장실', '직원화장실', '고객화장실', '방문자화장실',
        '세면실', '세면대', '파우더룸', '화장실습실',
        
        # Departments and organizational units
        '인사부', '인사팀', '인사과', '인적자원부', 'HR부서',
        '총무부', '총무팀', '총무과', '관리부', '사무관리팀',
        '회계부', '회계팀', '회계과', '경리부', '재무회계팀',
        '재무부', '재무팀', '재무과', '자금관리팀', '투자관리팀',
        '영업부', '영업팀', '영업과', '세일즈팀', '판매부',
        '마케팅부', '마케팅팀', '홍보부', '광고부', '브랜딩팀',
        '기획부', '기획팀', '전략기획팀', '사업기획팀', '경영기획실',
        '개발부', '개발팀', '기술개발팀', '제품개발팀', 'R&D팀',
        '연구개발부', '연구소', '기술연구소', '연구센터',
        '품질보증부', '품질관리팀', 'QA팀', 'QC팀', '검사팀',
        '고객서비스부', '고객지원팀', '콜센터', '헬프데스크',
        '기술지원부', '기술지원팀', '테크니컬서포트', 'IT지원팀',
        '교육부', '교육팀', '연수원', '교육센터', '인재개발팀',
        '법무부', '법무팀', '준법감시팀', '컴플라이언스팀',
        '감사부', '감사팀', '내부감사실', '리스크관리팀',
        '구매부', '구매팀', '조달팀', '자재관리팀', '물류팀',
        '생산부', '생산팀', '제조팀', '공장관리팀', '품질관리팀',
        '정보시스템부', 'IT부', '전산팀', '시스템관리팀',
        
        # Building and facility terms
        '본사', '지사', '지점', '영업소', '사업소', '출장소',
        '사옥', '오피스빌딩', '업무용빌딩', '상업용빌딩',
        '층', '1층', '2층', '지하층', '옥상층', '중간층',
        '동', '서동', '남동', '북동', '중앙동', '별관',
        '구역', '섹션', '존', '에리어', '블록', '윙',
        '복도', '홀웨이', '통로', '연결통로', '내부통로',
        '엘리베이터', '승강기', '에스컬레이터', '계단', '비상계단',
        '주차장', '지하주차장', '옥상주차장', '직원주차장',
        '옥상', '루프탑', '테라스', '발코니', '야외공간'
    ]
    
    @classmethod
    def get_all_english_spaces(cls):
        """Get all English space names"""
        return cls.ENGLISH_RESIDENTIAL + cls.ENGLISH_OFFICE
    
    @classmethod
    def get_all_korean_spaces(cls):
        """Get all Korean space names"""
        return cls.KOREAN_RESIDENTIAL + cls.KOREAN_OFFICE
    
    @classmethod
    def get_all_spaces(cls):
        """Get all space names in both languages"""
        return cls.get_all_english_spaces() + cls.get_all_korean_spaces()

class FloorPlanOCRDetector:
    """
    OCR detector specifically designed for floor plan text recognition.
    Detects space names and labels in multiple languages for both residential and office environments.
    """
    
    def __init__(self, languages=['en', 'ko']):
        """
        Initialize the OCR detector with support for multiple languages.
        
        Args:
            languages (list): List of language codes to support
                            Default: ['en', 'ko'] for English and Korean
        """
        self.languages = languages
        self.reader = None
        self.vocabulary = SpaceVocabulary()
        self._initialize_reader()
    
    def _initialize_reader(self):
        """Initialize the EasyOCR reader with specified languages."""
        try:
            logger.info(f"Initializing EasyOCR with languages: {self.languages}")
            self.reader = easyocr.Reader(self.languages, gpu=False)
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {str(e)}")
            self.reader = None
    
    def preprocess_image_for_ocr(self, image: np.ndarray, save_debug=False) -> np.ndarray:
        """
        Preprocess image to improve OCR accuracy for floor plan text.
        
        Args:
            image: Input image as numpy array
            save_debug: Boolean to save the preprocessed image for debugging
            
        Returns:
            Preprocessed image
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Apply contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Apply slight blur to reduce noise
            blurred = cv2.GaussianBlur(enhanced, (1, 1), 0)
            
            if save_debug:
                cv2.imwrite("ocr_preprocessed_debug.png", blurred)
            
            return blurred
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {str(e)}")
            return image
    
    def detect_text_regions(self, image: np.ndarray) -> List[Dict]:
        """
        Detect all text regions in the image using multiple approaches.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected text regions
        """
        if self.reader is None:
            logger.error("OCR reader not initialized")
            return []
        
        try:
            all_detections = []
            
            # Method 1: OCR on original image
            logger.info("Running OCR on original image...")
            try:
                results = self.reader.readtext(image)
                for (bbox, text, confidence) in results:
                    if text.strip() and confidence > 0.1:  # Very low threshold
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        bbox_formatted = [int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))]
                        all_detections.append({
                            'text': text.strip(),
                            'bbox': bbox_formatted,
                            'confidence': confidence,
                            'method': 'original'
                        })
            except Exception as e:
                logger.error(f"Error in original OCR: {str(e)}")
            
            # Method 2: OCR on preprocessed image
            logger.info("Running OCR on preprocessed image...")
            try:
                preprocessed = self.preprocess_image_for_ocr(image, save_debug=True)
                results = self.reader.readtext(preprocessed)
                for (bbox, text, confidence) in results:
                    if text.strip() and confidence > 0.1:  # Very low threshold
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        bbox_formatted = [int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))]
                        all_detections.append({
                            'text': text.strip(),
                            'bbox': bbox_formatted,
                            'confidence': confidence,
                            'method': 'preprocessed'
                        })
            except Exception as e:
                logger.error(f"Error in preprocessed OCR: {str(e)}")
            
            # Method 3: OCR with different parameters
            logger.info("Running OCR with enhanced parameters...")
            try:
                results = self.reader.readtext(image, 
                                            width_ths=0.2,
                                            height_ths=0.2,
                                            paragraph=False,  # Don't group into paragraphs
                                            detail=1)  # Return detailed results
                for (bbox, text, confidence) in results:
                    if text.strip() and confidence > 0.03:
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        bbox_formatted = [int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))]
                        all_detections.append({
                            'text': text.strip(),
                            'bbox': bbox_formatted,
                            'confidence': confidence,
                            'method': 'enhanced'
                        })
            except Exception as e:
                logger.error(f"Error in enhanced OCR: {str(e)}")
            
            # Method 5: OCR with very aggressive settings for small text
            logger.info("Running OCR with aggressive settings for small text...")
            try:
                results = self.reader.readtext(image, 
                                            width_ths=0.05,
                                            height_ths=0.05,
                                            paragraph=False,
                                            detail=1,
                                            min_size=3)  # Allow very small text
                for (bbox, text, confidence) in results:
                    if text.strip() and confidence > 0.01:  # Extremely low threshold
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        bbox_formatted = [int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))]
                        all_detections.append({
                            'text': text.strip(),
                            'bbox': bbox_formatted,
                            'confidence': confidence,
                            'method': 'aggressive'
                        })
            except Exception as e:
                logger.error(f"Error in aggressive OCR: {str(e)}")
            
            # Method 6: OCR on preprocessed and upscaled image for better small text detection
            logger.info("Running OCR on preprocessed and upscaled image...")
            try:
                preprocessed = self.preprocess_image_for_ocr(image, save_debug=False)
                upscaled_preprocessed = cv2.resize(preprocessed, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                results = self.reader.readtext(upscaled_preprocessed, 
                                            width_ths=0.1,
                                            height_ths=0.1,
                                            paragraph=False,
                                            detail=1)
                for (bbox, text, confidence) in results:
                    # More stringent filtering for upscaled results
                    if text.strip() and confidence > 0.1 and len(text.strip()) > 1:  # Require at least 2 characters
                        # Scale bbox coordinates back to original size
                        x_coords = [point[0] / 2.0 for point in bbox]
                        y_coords = [point[1] / 2.0 for point in bbox]
                        bbox_formatted = [int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))]
                        all_detections.append({
                            'text': text.strip(),
                            'bbox': bbox_formatted,
                            'confidence': confidence,
                            'method': 'preprocessed_upscaled'
                        })
            except Exception as e:
                logger.error(f"Error in preprocessed upscaled OCR: {str(e)}")
            
            # Remove duplicates with improved logic
            unique_detections = []
            
            # Sort by confidence (highest first) to keep the best detections
            all_detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            for det in all_detections:
                is_duplicate = False
                current_text = det['text'].lower().strip()
                current_bbox = det['bbox']
                current_center = ((current_bbox[0] + current_bbox[2]) / 2, (current_bbox[1] + current_bbox[3]) / 2)
                
                for existing in unique_detections:
                    existing_text = existing['text'].lower().strip()
                    existing_bbox = existing['bbox']
                    existing_center = ((existing_bbox[0] + existing_bbox[2]) / 2, (existing_bbox[1] + existing_bbox[3]) / 2)
                    
                    # Calculate distance between centers
                    distance = ((current_center[0] - existing_center[0])**2 + (current_center[1] - existing_center[1])**2)**0.5
                    
                    # Check for various types of duplicates
                    if distance < 50:  # Within 50 pixels
                        # Exact text match
                        if current_text == existing_text:
                            is_duplicate = True
                            break
                        
                        # One text is contained in the other (e.g., "BEDROOM" vs "BEDROOM 1")
                        if current_text in existing_text or existing_text in current_text:
                            is_duplicate = True
                            break
                        
                        # Similar text with small differences (OCR variations)
                        if len(current_text) > 3 and len(existing_text) > 3:
                            # Calculate text similarity
                            common_chars = sum(1 for a, b in zip(current_text, existing_text) if a == b)
                            similarity = common_chars / max(len(current_text), len(existing_text))
                            if similarity > 0.8:  # 80% similar
                                is_duplicate = True
                                break
                    
                    # Check for bbox overlap (even if centers are far apart)
                    overlap_x = max(0, min(current_bbox[2], existing_bbox[2]) - max(current_bbox[0], existing_bbox[0]))
                    overlap_y = max(0, min(current_bbox[3], existing_bbox[3]) - max(current_bbox[1], existing_bbox[1]))
                    overlap_area = overlap_x * overlap_y
                    
                    current_area = (current_bbox[2] - current_bbox[0]) * (current_bbox[3] - current_bbox[1])
                    existing_area = (existing_bbox[2] - existing_bbox[0]) * (existing_bbox[3] - existing_bbox[1])
                    
                    # If bboxes overlap significantly and text is similar
                    if overlap_area > 0:
                        overlap_ratio = overlap_area / min(current_area, existing_area)
                        if overlap_ratio > 0.5 and (current_text in existing_text or existing_text in current_text):
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    unique_detections.append(det)
            
            logger.info(f"Detected {len(unique_detections)} unique text regions from {len(all_detections)} total detections")
            return unique_detections
            
        except Exception as e:
            logger.error(f"Error in text detection: {str(e)}")
            return []
    
    def filter_space_names(self, detections: List[Dict], image_shape: Tuple[int, int]) -> List[Dict]:
        """
        Apply intelligent filtering to keep only likely space names using comprehensive vocabulary.
        
        Args:
            detections: List of text detections
            image_shape: Shape of the image (height, width)
            
        Returns:
            Filtered list of space names
        """
        img_h, img_w = image_shape
        filtered = []
        
        # Get comprehensive vocabulary
        all_english_spaces = self.vocabulary.get_all_english_spaces()
        all_korean_spaces = self.vocabulary.get_all_korean_spaces()
        
        # Patterns that indicate descriptive text (not room names)
        descriptive_patterns = [
            r'\d+\s*(bedrooms|bathrooms)',  # "4 Bedrooms", "3 Bathrooms" (plural only)
            r'(bedrooms|bathrooms)\s*/\s*\d+',  # "Bedrooms/3" (plural only)
            r'\d+\s*bed\s*/\s*\d+\s*bath',  # "4 bed/3 bath"
            r'(sq\s*ft|square\s*feet|sqft)',  # Square footage
            r'(floor\s*plan|house\s*plan|plan\s*view|office\s*plan)',  # Plan descriptions
            r'(scale|drawing|blueprint|architect)',  # Technical terms
            r'(total|approx|approximately)',  # Measurement terms
            r'(main\s*floor|upper\s*floor|lower\s*floor)',  # Floor descriptions
            r'(first\s*floor|second\s*floor|ground\s*floor)',  # Floor levels
            r'(lot\s*size|home\s*size|building\s*size|office\s*size)',  # Size descriptions
            r'(suite\s*\d+|unit\s*\d+|floor\s*\d+)',  # Suite/unit numbers
            r'(north|south|east|west|wing)',  # Directional descriptions
        ]
        
        # Enhanced patterns for valid room names (including fractions and abbreviations)
        valid_room_patterns = [
            # English fractional patterns
            r'^\d+/\d+\s*(bath|bathroom)$',  # "1/2 Bath", "3/4 Bathroom"
            r'^(half|quarter)\s*(bath|bathroom)$',  # "Half Bath", "Quarter Bathroom"
            
            # English compound room patterns
            r'^(powder|guest|master|en)\s*(room|bath|bathroom)$',  # "Powder Room", "Guest Bath"
            r'^(walk|walk-in)\s*(closet|pantry)$',  # "Walk-in Closet"
            r'^(conference|meeting|board)\s*(room|hall)$',  # "Conference Room"
            r'^(break|lunch|staff)\s*(room|area)$',  # "Break Room"
            r'^(server|storage|supply)\s*(room|closet)$',  # "Server Room"
            r'^(executive|private|corner)\s*(office|room)$',  # "Executive Office"
            
            # Korean patterns (common room names)
            r'.*?(주방|부엌)',  # Kitchen
            r'.*?(거실|응접실)',  # Living room
            r'.*?(침실|안방|방)',  # Bedroom
            r'.*?(화장실|욕실|변소)',  # Bathroom
            r'.*?(식당|다이닝)',  # Dining room
            r'.*?(서재|공부방|사무실)',  # Study/Office
            r'.*?(현관|입구)',  # Entrance
            r'.*?(베란다|발코니)',  # Balcony
            r'.*?(다용도실|세탁실)',  # Utility room
            r'.*?(창고|저장실)',  # Storage
            r'.*?(드레스룸|옷장)',  # Dressing room/Closet
            r'.*?(팬트리)',  # Pantry
            r'.*?(계단|층계)',  # Stairs
            r'.*?(회의실|미팅룸)',  # Meeting room
            r'.*?(휴게실|라운지)',  # Break room/Lounge
            r'.*?(임원실|대표실)',  # Executive office
        ]
        
        for det in detections:
            text = det['text'].strip()
            
            # Skip empty text
            if not text:
                continue
            
            # Skip single characters or very short text (likely OCR noise)
            if len(text.strip()) <= 1:
                logger.debug(f"Filtered single character: {text}")
                continue
            
            # Skip text with very low confidence unless it's a known room word
            confidence = det.get('confidence', 0)
            if confidence < 0.3:
                # Allow low confidence only for known room words
                known_room_words = ['formal', 'living', 'dining', 'kitchen', 'bedroom', 'bathroom', 'office', 'room']
                if not any(word in text.lower() for word in known_room_words):
                    logger.debug(f"Filtered low confidence non-room text: {text} (confidence: {confidence:.3f})")
                    continue
            
            # Skip text that's mostly non-alphanumeric (likely symbols or noise)
            alphanumeric_chars = sum(1 for c in text if c.isalnum())
            if alphanumeric_chars / len(text) < 0.5:  # Less than 50% alphanumeric
                logger.debug(f"Filtered non-alphanumeric text: {text}")
                continue
            
            # Skip obvious watermarks
            text_lower = text.lower()
            if any(watermark in text_lower for watermark in WATERMARK_KEYWORDS):
                logger.debug(f"Filtered watermark: {text}")
                continue
            
            # Skip standalone numbers (likely room numbers detected separately)
            if text.strip().isdigit() and len(text.strip()) <= 2:
                logger.debug(f"Filtered standalone number: {text}")
                continue
            
            # Check if it's a valid room name first (before filtering)
            is_valid_room = False
            
            # Check English patterns (case-insensitive)
            for pattern in valid_room_patterns:
                if re.search(pattern, text_lower):
                    logger.debug(f"Kept valid room pattern '{pattern}': {text}")
                    is_valid_room = True
                    break
            
            # Check comprehensive English vocabulary (exact and partial matches)
            if not is_valid_room:
                for space_name in all_english_spaces:
                    space_lower = space_name.lower()
                    # Exact match
                    if text_lower == space_lower:
                        logger.debug(f"Kept exact English match '{space_name}': {text}")
                        is_valid_room = True
                        break
                    # Partial match (space name contains the text or vice versa)
                    elif (space_lower in text_lower and len(space_lower) > 2) or \
                         (text_lower in space_lower and len(text_lower) > 2):
                        logger.debug(f"Kept partial English match '{space_name}': {text}")
                        is_valid_room = True
                        break
            
            # Check comprehensive Korean vocabulary (case-sensitive for Korean)
            if not is_valid_room:
                for space_name in all_korean_spaces:
                    # Exact match
                    if text == space_name:
                        logger.debug(f"Kept exact Korean match '{space_name}': {text}")
                        is_valid_room = True
                        break
                    # Partial match (space name contains the text or vice versa)
                    elif (space_name in text and len(space_name) > 1) or \
                         (text in space_name and len(text) > 1):
                        logger.debug(f"Kept partial Korean match '{space_name}': {text}")
                        is_valid_room = True
                        break
            
            # If it's a valid room, skip all other filters
            if not is_valid_room:
                # Skip descriptive patterns
                is_descriptive = False
                for pattern in descriptive_patterns:
                    if re.search(pattern, text_lower):
                        logger.debug(f"Filtered descriptive pattern '{pattern}': {text}")
                        is_descriptive = True
                        break
                
                if is_descriptive:
                    continue
                
                # Skip text with numbers and room counts (like "4 Bedrooms/3 Bathrooms")
                if re.search(r'\d+.*\d+', text) and any(word in text_lower for word in ['bedrooms', 'bathrooms', 'offices', 'rooms']):
                    logger.debug(f"Filtered room count: {text}")
                    continue
                
                # Skip common office building descriptors
                office_descriptors = [
                    'floor plan', 'office plan', 'building plan', 'layout',
                    'suite', 'level', 'floor', 'wing', 'zone', 'area',
                    'north', 'south', 'east', 'west', 'central',
                    'entrance', 'exit', 'corridor', 'hallway', 'lobby',
                    'elevator', 'stairs', 'emergency', 'fire exit'
                ]
                if any(descriptor in text_lower for descriptor in office_descriptors) and not is_valid_room:
                    logger.debug(f"Filtered office descriptor: {text}")
                    continue
            
            # Skip very long text (likely descriptions or titles)
            # Korean text can be more compact, so adjust length limits
            max_length = 20 if any('\u3130' <= c <= '\u318F' or '\uAC00' <= c <= '\uD7A3' for c in text) else 30
            if len(text) > max_length:
                logger.debug(f"Filtered long text: {text}")
                continue
            
            # Skip text with multiple words that might be descriptive
            # Korean text doesn't use spaces the same way, so be more lenient
            words = text.split()
            has_korean = any('\u3130' <= c <= '\u318F' or '\uAC00' <= c <= '\uD7A3' for c in text)
            max_words = 5 if has_korean else 3  # Allow more "words" for Korean mixed text
            if len(words) > max_words:
                logger.debug(f"Filtered multi-word text: {text}")
                continue
            
            # Skip text that's too small or too large
            x1, y1, x2, y2 = det['bbox']
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            if area < 20:  # Too small
                logger.debug(f"Filtered too small: {text}")
                continue
                
            if area > 8000:  # Reduced from 10000 to catch more descriptive text
                logger.debug(f"Filtered too large: {text}")
                continue
            
            # Skip text in the top 20% of the image (likely titles/descriptions)
            center_y = (y1 + y2) / 2
            if center_y < img_h * 0.20:  # Increased from 0.15 to catch more titles
                logger.debug(f"Filtered title area: {text}")
                continue
            
            # Skip text in the bottom 15% of the image (likely legends)
            if center_y > img_h * 0.85:  # Increased from 0.90 to catch more legends
                logger.debug(f"Filtered legend area: {text}")
                continue
            
            # Skip text in the left and right margins (5% each side)
            center_x = (x1 + x2) / 2
            if center_x < img_w * 0.05 or center_x > img_w * 0.95:
                logger.debug(f"Filtered margin text: {text}")
                continue
            
            # Keep everything else
            filtered.append(det)
            logger.debug(f"Kept space name: {text}")
        
        logger.info(f"Filtered to {len(filtered)} space names from {len(detections)} detections")
        return filtered
    
    def _clean_room_name(self, text: str) -> str:
        """
        Clean up room name text by removing common OCR artifacts and extra characters.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned room name
        """
        # Remove extra whitespace
        cleaned = text.strip()
        
        # Filter out corrupted Unicode text
        if len(cleaned) < 10:  # Short strings with corrupted Unicode
            # Check for corrupted Korean Unicode sequences like \uc624h
            if '\\u' in cleaned or any(ord(c) > 0x1F000 for c in cleaned):
                logger.debug(f"Filtering out corrupted Unicode text: '{cleaned}'")
                return ""
        
        # Remove common OCR artifacts
        cleaned = re.sub(r'[^\w\s/\-가-힣]', '', cleaned)  # Keep alphanumeric, spaces, slashes, hyphens, Korean
        
        # Remove isolated single characters that are likely OCR noise
        words = cleaned.split()
        valid_words = []
        for word in words:
            # Keep words that are at least 2 characters, or single characters that are numbers
            if len(word) >= 2 or word.isdigit():
                valid_words.append(word)
            # Also keep single letters if they're part of common abbreviations
            elif word.lower() in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']:
                # Only keep if it's not isolated noise
                if len(words) > 1:
                    valid_words.append(word)
        
        cleaned = ' '.join(valid_words)
        
        # Remove trailing numbers that might be room numbers detected as part of the name
        # But keep fractions like "1/2" and numbered rooms like "Bath1", "Bath2"
        if not re.search(r'\d+/\d+', cleaned):  # If it's not a fraction
            # Don't remove numbers that are part of room names (like Bath1, Bath2)
            if not re.search(r'(bath|bedroom|room)\s*\d+$', cleaned.lower()):
                cleaned = re.sub(r'\s+\d+$', '', cleaned)  # Remove trailing numbers
        
        # Clean up multiple spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Capitalize properly for English text
        if not any('\u3130' <= c <= '\u318F' or '\uAC00' <= c <= '\uD7A3' for c in cleaned):  # If not Korean
            cleaned = cleaned.title()
        
        return cleaned.strip()

    def _deduplicate_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Remove duplicate detections of the same text at the same location.
        
        Args:
            detections: List of text detections
            
        Returns:
            List of deduplicated text detections
        """
        if not detections:
            return detections
        
        deduplicated = []
        used_indices = set()
        
        for i, detection in enumerate(detections):
            if i in used_indices:
                continue
                
            current_text = detection['text'].strip().lower()
            current_bbox = detection['bbox']
            current_center = ((current_bbox[0] + current_bbox[2]) / 2, (current_bbox[1] + current_bbox[3]) / 2)
            
            # Skip corrupted or invalid text
            if len(current_text) < 2 or not current_text.replace(' ', '').replace('-', '').replace('/', '').isalnum():
                # Check if it's valid Korean
                is_korean = any('\uAC00' <= c <= '\uD7A3' for c in current_text)
                if not is_korean:
                    logger.debug(f"Skipping corrupted text: '{current_text}'")
                    used_indices.add(i)
                    continue
            
            # Find duplicates of this detection
            candidates_to_merge = [detection]
            
            for j, other_detection in enumerate(detections):
                if i == j or j in used_indices:
                    continue
                    
                other_text = other_detection['text'].strip().lower()
                other_bbox = other_detection['bbox']
                other_center = ((other_bbox[0] + other_bbox[2]) / 2, (other_bbox[1] + other_bbox[3]) / 2)
                
                # Check if this is a duplicate (same location, similar text)
                distance = ((current_center[0] - other_center[0]) ** 2 + (current_center[1] - other_center[1]) ** 2) ** 0.5
                
                if distance < 30:  # Very close locations (within 30 pixels)
                    # Check text similarity
                    is_duplicate = False
                    
                    # Exact match
                    if current_text == other_text:
                        is_duplicate = True
                    
                    # One is substring of the other (e.g., "Bath" vs "Bath1")
                    elif current_text in other_text or other_text in current_text:
                        is_duplicate = True
                    
                    # OCR errors (e.g., "Laundry" vs "Lauivdry") 
                    elif len(current_text) > 3 and len(other_text) > 3:
                        # Simple character similarity check
                        common_chars = sum(1 for a, b in zip(current_text, other_text) if a == b)
                        similarity = common_chars / max(len(current_text), len(other_text))
                        if similarity > 0.7:  # 70% character similarity
                            is_duplicate = True
                    
                    if is_duplicate:
                        candidates_to_merge.append(other_detection)
                        used_indices.add(j)
                        logger.debug(f"Found duplicate: '{current_text}' and '{other_text}'")
            
            # Choose the best detection from duplicates
            if candidates_to_merge:
                # Sort by confidence, then by text quality (longer text usually better)
                best_detection = max(candidates_to_merge, 
                                   key=lambda x: (x['confidence'], len(x['text'].strip())))
                
                # Log what we're keeping vs discarding
                if len(candidates_to_merge) > 1:
                    kept_text = best_detection['text'].strip()
                    discarded_texts = [c['text'].strip() for c in candidates_to_merge if c != best_detection]
                    logger.info(f"Deduplicated: kept '{kept_text}', discarded {discarded_texts}")
                
                deduplicated.append(best_detection)
            
            used_indices.add(i)
        
        logger.info(f"Deduplicated {len(detections)} detections into {len(deduplicated)} unique detections")
        return deduplicated

    def _merge_nearby_text(self, detections: List[Dict]) -> List[Dict]:
        """
        Merge nearby text detections that should be combined into single room names.
        
        Args:
            detections: List of text detections
            
        Returns:
            List of merged text detections
        """
        if not detections:
            return detections
        
        merged = []
        used_indices = set()
        
        # Sort detections by position (left to right, top to bottom)
        sorted_detections = sorted(detections, key=lambda x: (x['bbox'][1], x['bbox'][0]))
        
        for i, detection in enumerate(sorted_detections):
            if i in used_indices:
                continue
                
            current_text = detection['text'].strip()
            current_bbox = detection['bbox']
            current_center = ((current_bbox[0] + current_bbox[2]) / 2, (current_bbox[1] + current_bbox[3]) / 2)
            
            # Skip single character detections for merging (likely noise)
            if len(current_text) <= 1:
                merged.append(detection)
                used_indices.add(i)
                continue
            
            # Look for nearby text that should be merged
            candidates_to_merge = [detection]
            
            for j, other_detection in enumerate(sorted_detections):
                if i == j or j in used_indices:
                    continue
                    
                other_text = other_detection['text'].strip()
                other_bbox = other_detection['bbox']
                other_center = ((other_bbox[0] + other_bbox[2]) / 2, (other_bbox[1] + other_bbox[3]) / 2)
                
                # Skip single character detections for merging (likely noise)
                if len(other_text) <= 1:
                    continue
                
                # Check if texts should be merged based on proximity and content
                should_merge = False
                
                # Check for horizontal proximity (same line)
                vertical_distance = abs(current_center[1] - other_center[1])
                horizontal_distance = abs(current_center[0] - other_center[0])
                
                # Reasonable search distance for merging
                max_vertical_distance = 30  # Same line tolerance
                max_horizontal_distance = 150  # Reasonable horizontal distance
                
                if vertical_distance < max_vertical_distance and horizontal_distance < max_horizontal_distance:
                    # Check for compound room name patterns
                    combined_texts = [current_text.lower(), other_text.lower()]
                    combined_texts.sort()  # Sort to handle order independence
                    
                    # Common compound room name patterns
                    compound_patterns = [
                        ['eat-in', 'kitchen'], ['eat', 'kitchen'],
                        ['formal', 'living'], ['formal', 'dining'],
                        ['dining', 'room'], ['living', 'room'], ['family', 'room'],
                        ['guest', 'room'], ['master', 'bedroom'], ['master', 'bath'],
                        ['1/2', 'bath'], ['half', 'bath'], ['3/4', 'bath'],
                        ['powder', 'room'], ['laundry', 'room'], ['utility', 'room'],
                        ['walk-in', 'closet'], ['home', 'office'], ['great', 'room']
                    ]
                    
                    # Check if the combination matches any compound pattern
                    for pattern in compound_patterns:
                        if combined_texts == pattern:
                            should_merge = True
                            logger.info(f"Merging compound pattern: {combined_texts}")
                            break
                    
                    # Special handling for "formal" as a modifier
                    if not should_merge and 'formal' in combined_texts:
                        other_word = [word for word in combined_texts if word != 'formal'][0] if len(combined_texts) == 2 else None
                        if other_word and other_word in ['living', 'dining', 'room', 'lounge']:
                            should_merge = True
                            logger.info(f"Merging formal space: {combined_texts}")
                    
                    # Check for generic patterns like "word room"
                    if not should_merge:
                        for text in combined_texts:
                            if text in ['room', 'office', 'area', 'space', 'closet', 'bath', 'bathroom']:
                                other_text_clean = [t for t in combined_texts if t != text][0] if len(combined_texts) == 2 else None
                                if other_text_clean and len(other_text_clean) > 2:  # Reasonable word length
                                    should_merge = True
                                    logger.info(f"Merging generic pattern: {combined_texts}")
                                    break
                
                if should_merge:
                    candidates_to_merge.append(other_detection)
                    used_indices.add(j)
            
            # Merge the candidates
            if len(candidates_to_merge) > 1:
                # Sort candidates by x-coordinate (left to right)
                candidates_to_merge.sort(key=lambda x: x['bbox'][0])
                
                # Combine texts
                merged_text = ' '.join([candidate['text'].strip() for candidate in candidates_to_merge])
                
                # Calculate combined bounding box
                all_x1 = [candidate['bbox'][0] for candidate in candidates_to_merge]
                all_y1 = [candidate['bbox'][1] for candidate in candidates_to_merge]
                all_x2 = [candidate['bbox'][2] for candidate in candidates_to_merge]
                all_y2 = [candidate['bbox'][3] for candidate in candidates_to_merge]
                
                merged_bbox = [min(all_x1), min(all_y1), max(all_x2), max(all_y2)]
                
                # Use the highest confidence
                merged_confidence = max([candidate['confidence'] for candidate in candidates_to_merge])
                
                # Combine methods
                methods = list(set([candidate.get('method', 'unknown') for candidate in candidates_to_merge]))
                merged_method = '+'.join(methods)
                
                merged_detection = {
                    'text': merged_text,
                    'bbox': merged_bbox,
                    'confidence': merged_confidence,
                    'method': merged_method
                }
                
                merged.append(merged_detection)
                logger.info(f"Merged texts: {[c['text'] for c in candidates_to_merge]} -> '{merged_text}'")
            else:
                merged.append(detection)
            
            used_indices.add(i)
        
        logger.info(f"Merged {len(detections)} detections into {len(merged)} combined detections")
        return merged

    def get_space_names(self, image: np.ndarray, wall_bboxes: Optional[List[List[int]]] = None) -> List[Dict]:
        """
        Main method to detect space names in a floor plan image.
        
        Args:
            image: Input floor plan image as numpy array
            wall_bboxes: List of wall bounding boxes [x1, y1, x2, y2] (optional)
            
        Returns:
            List of dictionaries containing space name information
        """
        try:
            # Detect all text regions
            text_regions = self.detect_text_regions(image)
            
            # First, remove duplicates (same text at same location from different detection methods)
            deduplicated_regions = self._deduplicate_detections(text_regions)
            
            # Then merge nearby text that should be combined (e.g., "Formal" + "Dining")
            merged_regions = self._merge_nearby_text(deduplicated_regions)
            
            # Apply filtering to merged regions
            filtered_regions = self.filter_space_names(merged_regions, image.shape[:2])
            
            # Format as space names with text cleanup
            space_names = []
            space_counter = 1
            for region in filtered_regions:
                # Clean up the text
                cleaned_text = self._clean_room_name(region['text'])
                
                # Skip if text becomes empty after cleaning
                if not cleaned_text or len(cleaned_text.strip()) == 0:
                    logger.debug(f"Skipping empty text after cleaning: '{region['text']}'")
                    continue
                
                space_name = {
                    'id': f'SPACE_{space_counter}',
                    'name': cleaned_text,
                    'bbox': region['bbox'],
                    'confidence': region['confidence'],
                    'method': region.get('method', 'unknown'),
                    'center': {
                        'x': (region['bbox'][0] + region['bbox'][2]) / 2,
                        'y': (region['bbox'][1] + region['bbox'][3]) / 2
                    }
                }
                space_names.append(space_name)
                space_counter += 1
            
            logger.info(f"Final result: {len(space_names)} space names detected")
            return space_names
            
        except Exception as e:
            logger.error(f"Error in space name detection: {str(e)}")
            return []
    
    def detect_text_in_region(self, image: np.ndarray, region_bbox: List[int]) -> List[Dict]:
        """
        Detect text within a specific region of the image.
        
        Args:
            image: Input image
            region_bbox: Bounding box [x1, y1, x2, y2] of the region to analyze
            
        Returns:
            List of text detections within the region
        """
        try:
            x1, y1, x2, y2 = region_bbox
            
            # Ensure coordinates are within image bounds
            img_h, img_w = image.shape[:2]
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(x1, min(x2, img_w))
            y2 = max(y1, min(y2, img_h))
            
            # Extract region
            region_image = image[y1:y2, x1:x2]
            
            # Detect text in region
            text_regions = self.detect_text_regions(region_image)
            
            # Adjust coordinates back to original image
            for region in text_regions:
                region['bbox'][0] += x1
                region['bbox'][1] += y1
                region['bbox'][2] += x1
                region['bbox'][3] += y1
            
            return text_regions
            
        except Exception as e:
            logger.error(f"Error in region text detection: {str(e)}")
            return []

# Global OCR detector instance
_ocr_detector = None

def get_ocr_detector() -> FloorPlanOCRDetector:
    """
    Get or create the global OCR detector instance.
    
    Returns:
        FloorPlanOCRDetector instance
    """
    global _ocr_detector
    if _ocr_detector is None:
        _ocr_detector = FloorPlanOCRDetector()
    return _ocr_detector

def detect_space_names(image: np.ndarray, wall_bboxes: Optional[List[List[int]]] = None) -> List[Dict]:
    """
    Convenience function to detect space names in an image.
    
    Args:
        image: Input image as numpy array
        wall_bboxes: List of wall bounding boxes [x1, y1, x2, y2] (optional)
        
    Returns:
        List of space name dictionaries
    """
    detector = get_ocr_detector()
    return detector.get_space_names(image, wall_bboxes=wall_bboxes)

def detect_text_in_region(image: np.ndarray, region_bbox: List[int]) -> List[Dict]:
    """
    Convenience function to detect text in a specific region.
    
    Args:
        image: Input image as numpy array
        region_bbox: Bounding box [x1, y1, x2, y2] of the region
        
    Returns:
        List of text detections within the region
    """
    detector = get_ocr_detector()
    return detector.detect_text_in_region(image, region_bbox) 