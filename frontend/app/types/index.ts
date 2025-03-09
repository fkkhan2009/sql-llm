export enum StreamResponseType {
    STEP_START = 'step_start',
    STEP_END = 'step_end',
    LLM_CHUNK = 'llm_chunk',
    RESULT = 'result',
    ERROR = 'error',
  }
  
  export enum ProcessStep {
    CONTEXT = 'context',
    TABLES = 'tables',
    SCHEMA = 'schema',
    SQL = 'sql',
    EXECUTION = 'execution',
    EXPLANATION = 'explanation',
  }
  
  export interface StreamResponse {
    type: StreamResponseType;
    step: ProcessStep;
    content?: any;
    message?: string;
    clause?: string;
  }