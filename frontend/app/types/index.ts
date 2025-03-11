export enum StreamResponseType {
    STEP_START = 'step_start',
    STEP_END = 'step_end',
    LLM_CHUNK = 'llm_chunk',
    RESULT = 'result',
    ERROR = 'error',
    QUERY_COMPLETE = 'query_complete',
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

export interface ChatMessage {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
    metadata?: {
        sql?: string;
        results?: any;
        explanation?: string;
        steps?: Map<ProcessStep, {
            status: 'pending' | 'in-progress' | 'completed' | 'error';
            message?: string;
            llmOutput?: string;
        }>;
    };
}