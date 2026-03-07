import { EventEmitter } from "events";

interface EventMap {
  prefsUpdated: [];
}

class TypedEventEmitter extends EventEmitter {
  override emit<K extends keyof EventMap>(event: K, ...args: EventMap[K]): boolean {
    return super.emit(event, ...args);
  }

  override on<K extends keyof EventMap>(event: K, listener: (...args: EventMap[K]) => void): this {
    return super.on(event, listener as (...args: unknown[]) => void);
  }
}

export const eventBus = new TypedEventEmitter();
